from scipy.spatial import Voronoi
import itertools
from PIL import Image, ImageDraw
from scipy.ndimage import map_coordinates, distance_transform_edt, gaussian_filter
import scipy as sp
import numpy as np
from procedural_driving.procedural import perlin
import yaml
from typing import Tuple
import os

RANDOM_POINTS_SEED = 1
WARP_SEED = 2


def chunks_around(i, j):
    return itertools.product(range(i - 1, i + 2), range(j - 1, j + 2))


def generate_random_points_for_chunk(i: int, j: int, seed: int, n: int, size: int):
    np.random.seed(((i * size**2 + j * size) + seed + RANDOM_POINTS_SEED) % (2**32))
    points = np.random.random(size=(n, 2)) * 100
    return points


def warp_image(
    image, i: int, j: int, seed: int, size: int, noise_levels={4: 30, 16: 10}
) -> np.ndarray:
    """
    Warp an image by adding perlin noise to the coordinates.

    Used to make the mountains look more natural.
    """

    # Random warp by adding perlin noise to the coordinates
    np.random.seed(((i * size**2 + j * size) + seed + WARP_SEED) % (2**32))
    x, y = np.meshgrid(np.arange(size, size * 2 + 1), np.arange(size, size * 2 + 1))
    x = x + perlin.perlin_chunk(i, j, size + 1, noise_levels, 0)
    y = y + perlin.perlin_chunk(i, j, size + 1, noise_levels, 1)
    warped_image = map_coordinates(image, [y, x], order=1)
    return warped_image


_config = None


def default_config() -> dict:
    global _config
    if _config is None:
        _config = yaml.load(
            open("configs/worlds/default.yaml", "r"), Loader=yaml.FullLoader
        )
    return _config


def set_config_name(config_name: str) -> dict:
    global _config
    _config = yaml.load(
        open(os.path.join(os.path.dirname(__file__), f"../../configs/worlds/{config_name}.yaml"), "r"), Loader=yaml.FullLoader
    )


def gen_terrain(
    i: int, j: int, seed: int, size: int, config: dict = None
) -> Tuple[np.ndarray, dict]:
    """
    Generate terrain heightmap for a chunk.
    """

    if config is None:
        config = default_config()["terrain"]
    elif isinstance(config, str):
        global _config
        _config = yaml.load(open(config, "r"), Loader=yaml.FullLoader)["terrain"]
        config = _config

    mountains_per_chunk = config["mountains_per_chunk"]
    mountain_scale = config["mountain_scale"]
    warp_noise_levels = {k: v * size for k, v in config["warp_noise_levels"].items()}
    terrain_noise_levels = config["terrain_noise_levels"]

    if mountains_per_chunk > 0:
        points = np.concatenate(
            [
                np.array([size * (ip - i + 1), size * (jp - j + 1)])[None]
                + generate_random_points_for_chunk(
                    ip, jp, seed, mountains_per_chunk, size
                )
                for (ip, jp) in chunks_around(i, j)
            ],
            axis=0,
        )

        grid = np.zeros((size * 3, size * 3))

        # Draw the edges of the voronoi diagram
        vor = Voronoi(points)
        image = Image.fromarray(grid)
        draw = ImageDraw.Draw(image)
        for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
            simplex = np.asarray(simplex)
            if np.all(simplex >= 0):
                v0 = vor.vertices[simplex[0]]
                v1 = vor.vertices[simplex[1]]
                draw.line(
                    [(int(v0[1]), int(v0[0])), (int(v1[1]), int(v1[0]))],
                    fill=255,
                    width=int(np.round(size / 20)),
                )

        # Take the EDT
        grid = np.asarray(image)
        grid = distance_transform_edt(255 - grid) / size
        grid = gaussian_filter(grid, sigma=size / 300)
        # 7 is a magic number where sigmoid(1/(1+exp(-7x)) is roughly uniformly distributed for x ~ perlin
        # Less than 7 is unimodal, anything more is bimodal
        magic = 5.0
        mountain_scale_logit = perlin.perlin_frac_scale(i, j, size + 1, {3: 5.0}, seed)
        grid_mountain = (
            warp_image(grid, i, j, seed, size, noise_levels=warp_noise_levels)
            * sp.special.expit(mountain_scale_logit)
            * mountain_scale
        )
    else:
        grid_mountain = np.zeros((size + 1, size + 1))
        mountain_scale_logit = np.full((size + 1, size + 1), -100)

    # Add perlin noise
    grid = grid_mountain + perlin.perlin_chunk(
        i, j, size + 1, terrain_noise_levels, seed
    )
    grid = (grid + 0.75) / 2
    grid = np.clip(grid, 0, 1)
    return grid, {"mountain": grid_mountain, "mountain_scale": mountain_scale_logit}


def gen_texture(
    i: int, j: int, seed: int, size: int, config: dict = None, terrain_aux: dict = None
) -> np.ndarray:
    """
    Generate a texture for a chunk.

    Uses a combination of the mountain information and the perlin noise to generate a texture,
    with colors determined by the biome and high-frequency noise from an additional perlin noise layer.
    """

    if config is None:
        config = default_config()["visual"]
    elif isinstance(config, str):
        global _config
        _config = yaml.load(open(config, "r"), Loader=yaml.FullLoader)["visual"]
        config = _config
    
    if config is None:
        return None

    res = terrain_aux["mountain"].shape[0]
    highlands_selector = sp.ndimage.zoom(
        terrain_aux["mountain"], ((size + 1) / res, (size + 1) / res), order=1
    )
    mountain_scale = sp.ndimage.zoom(
        terrain_aux["mountain_scale"], ((size + 1) / res, (size + 1) / res), order=1
    )

    biome_selector = sp.special.expit(
        perlin.perlin_frac_scale(i, j, size + 1, {config["biome_scale"]: 5.0}, seed)
        + mountain_scale
    )

    texture = perlin.perlin_chunk(i, j, size + 1, {32: 0.5, 64: 0.5}, 1) + 0.5
    texture = texture + np.random.normal(0, 0.1, size=texture.shape)
    texture = np.clip(texture, 0, 1)[..., None]

    highlands_mux = (
        0.5 - np.tanh(15 * (0.05 - highlands_selector[..., None]) + 2 * texture) / 2
    )
    biome_a = config["biomes"][0]
    biome_b = config["biomes"][1]

    color_1_low = (
        biome_selector[..., None] * biome_a["lowlands_color_1"]
        + (1 - biome_selector[..., None]) * biome_b["lowlands_color_1"]
    )
    color_2_low = (
        biome_selector[..., None] * biome_a["lowlands_color_2"]
        + (1 - biome_selector[..., None]) * biome_b["lowlands_color_2"]
    )
    color_1_high = (
        biome_selector[..., None] * biome_a["highlands_color_1"]
        + (1 - biome_selector[..., None]) * biome_b["highlands_color_1"]
    )
    color_2_high = (
        biome_selector[..., None] * biome_a["highlands_color_2"]
        + (1 - biome_selector[..., None]) * biome_b["highlands_color_2"]
    )

    color_1 = highlands_mux * color_1_high + (1 - highlands_mux) * color_1_low
    color_2 = highlands_mux * color_2_high + (1 - highlands_mux) * color_2_low

    return np.flip(texture * color_1 + (1 - texture) * color_2, axis=1)
