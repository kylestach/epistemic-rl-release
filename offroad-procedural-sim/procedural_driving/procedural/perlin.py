# Perlin noise implementation
import numpy as np
from typing import Dict, Tuple


def perlin_chunk_vectors_single_scale(
    chunk_x: int, chunk_y: int, noise_scale: int, seed: int
) -> np.ndarray:
    """
    Generate a single scale of Perlin noise vectors for a chunk.
    """

    # Generate random vectors, one for each grid point
    # The vectors should be only generated using the seed, coordinates, and noise scale
    # This way, the same noise will be generated for the same seed, coordinates, and noise scale
    grid_x, grid_y = np.meshgrid(
        np.arange(chunk_x * noise_scale, (chunk_x + 1) * noise_scale + 1),
        np.arange(chunk_y * noise_scale, (chunk_y + 1) * noise_scale + 1),
    )

    # Custom PRNG so that the same seed will always generate the same angles, even on the edges
    a = grid_x
    b = grid_y

    a *= 3284157443
    b ^= (a << 16 | a >> 16) & 0xFFFFFFFF
    b *= 1911520717
    a ^= (b << 16 | b >> 16) & 0xFFFFFFFF
    a *= 2048419325
    a &= 0xFFFFFFFF
    seed ^= (a << 16 | a >> 16) & 0xFFFFFFFF
    seed *= 1911520717
    a ^= (seed << 16 | seed >> 16) & 0xFFFFFFFF
    a *= 2048419325
    a &= 0xFFFFFFFF
    angles = a * (2 * 3.14159265 / 0xFFFFFFFF)

    return np.stack([np.cos(angles), np.sin(angles)], axis=-1)


def smooth_interp_1d(a0: np.ndarray, a1: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Smoothly interpolate between two values.
    """
    return (a1 - a0) * ((w * (w * 6.0 - 15.0) + 10.0) * w * w * w) + a0


def smooth_interp_2d(
    v00: np.ndarray,
    v10: np.ndarray,
    v01: np.ndarray,
    v11: np.ndarray,
    wx: np.ndarray,
    wy: np.ndarray,
):
    """
    Smoothly interpolate a 2d grid of values.
    """
    return smooth_interp_1d(
        smooth_interp_1d(v00, v10, wx), smooth_interp_1d(v01, v11, wx), wy
    )


def smooth_interp_vectors(
    chunk_vectors: np.ndarray,
    interp_size: int,
    grid_min: Tuple[float, float] = (0, 0),
    grid_max: Tuple[float, float] = (1, 1),
) -> np.ndarray:
    """
    Smoothly interpolate a grid of Perlin vectors to a chunk of values.

    If grid_min and grid_max are specified to anything other than the range [0, 1],
    then the chunk will be sampled from the given subset block of the grid.
    """

    # Find the corresponding indices in the chunk vectors for each point in the interpolated chunk
    chunk_size = chunk_vectors.shape[0] - 1
    x_grid, y_grid = np.meshgrid(
        np.linspace(chunk_size * grid_min[0], chunk_size * grid_max[0], interp_size),
        np.linspace(chunk_size * grid_min[1], chunk_size * grid_max[1], interp_size),
    )
    x_indices = np.minimum(np.floor(x_grid).astype(int), chunk_size - 1)
    y_indices = np.minimum(np.floor(y_grid).astype(int), chunk_size - 1)

    # Find the vectors for each point in the interpolated chunk
    vectors_00 = chunk_vectors[x_indices, y_indices]
    vectors_10 = chunk_vectors[x_indices + 1, y_indices]
    vectors_01 = chunk_vectors[x_indices, y_indices + 1]
    vectors_11 = chunk_vectors[x_indices + 1, y_indices + 1]

    # Smooth interpolation
    wxs = x_grid - x_indices
    wys = y_grid - y_indices
    v00 = np.sum(vectors_00 * np.stack([wxs, wys], axis=-1), axis=-1)
    v10 = np.sum(vectors_10 * np.stack([wxs - 1, wys], axis=-1), axis=-1)
    v01 = np.sum(vectors_01 * np.stack([wxs, wys - 1], axis=-1), axis=-1)
    v11 = np.sum(vectors_11 * np.stack([wxs - 1, wys - 1], axis=-1), axis=-1)

    return smooth_interp_2d(v00, v10, v01, v11, wxs, wys)


def perlin_chunk(
    x: int, y: int, interp_size: int, noise_scales: Dict[int, float], seed: int
) -> np.ndarray:
    """
    Generate Perlin noise at regular octaves.
    """
    result = np.zeros((interp_size, interp_size))
    for noise_scale, magnitude in noise_scales.items():
        # Generate a chunk of perlin noise
        chunk_scaled_vectors = perlin_chunk_vectors_single_scale(
            x, y, noise_scale, seed
        )
        result += magnitude * smooth_interp_vectors(chunk_scaled_vectors, interp_size)
    return result


def perlin_frac_scale(
    x: int, y: int, interp_size: int, inverse_noise_scales: Dict[int, float], seed: int
) -> np.ndarray:
    """
    Generate perlin noise at **fractional** (<1) octaves.

    This is used for generating noise at larger-than-chunk scales, e.g. for biomes.
    """
    result = np.zeros((interp_size, interp_size))
    for inverse_noise_scale, magnitude in inverse_noise_scales.items():
        # Generate a chunk of perlin noise
        megachunk_x = np.floor(x / inverse_noise_scale).astype(int)
        megachunk_y = np.floor(y / inverse_noise_scale).astype(int)
        minichunk_x = x - megachunk_x * inverse_noise_scale
        minichunk_y = y - megachunk_y * inverse_noise_scale
        chunk_scaled_vectors = perlin_chunk_vectors_single_scale(
            megachunk_x, megachunk_y, 1, seed
        )
        grid_min = (
            minichunk_y / inverse_noise_scale,
            minichunk_x / inverse_noise_scale,
        )
        grid_max = (
            (minichunk_y + 1) / inverse_noise_scale,
            (minichunk_x + 1) / inverse_noise_scale,
        )
        result += magnitude * smooth_interp_vectors(
            chunk_scaled_vectors, interp_size, grid_min=grid_min, grid_max=grid_max
        )
    return result
