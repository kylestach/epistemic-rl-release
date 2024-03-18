from dm_control import mjcf
from dm_control.mjcf.element import RootElement
from dm_control import composer
from dm_control.mujoco.wrapper import mjbindings
from dataclasses import dataclass
import numpy as np
from procedural_driving.procedural import procedural_generation
from dm_control.mjcf import Physics
from typing import Dict, Optional

mjlib = mjbindings.mjlib
mjr_upload_texture = getattr(mjlib, 'mjr_uploadTexture')
mjr_upload_hfield = getattr(mjlib, 'mjr_uploadHField')


@dataclass(frozen=True)
class ChunkKey:
    """Dataclass for storing chunk indices."""

    pos: tuple
    seed: int


@dataclass
class ChunkData:
    """Dataclass for storing chunk data."""

    terrain: np.ndarray
    texture: Optional[np.ndarray]
    aux: dict


def _gen_chunk_terrain(chunk: ChunkKey, res: int, texture_res: int):
    """
    Helper function to generate terrain and texture data for a chunk.
    """
    i, j = chunk.pos
    seed = chunk.seed

    terrain, terrain_aux = procedural_generation.gen_terrain(i, j, seed, res - 1)

    texture = procedural_generation.gen_texture(
        i, j, seed, texture_res, terrain_aux=terrain_aux
    )

    return ChunkData(
        terrain=terrain,
        texture=texture[:-1, :-1] * 255 if texture is not None else None,
        aux=terrain_aux,
    )


class Chunk(composer.Entity):
    _chunk_data: Optional[ChunkData]
    _chunk_pos: Optional[tuple]

    def _build(
        self,
        idx: int,
        xyscale: float = 10.0,
        zscale: float = 5.0,
    ):
        self._mjcf_root = RootElement(model=f"chunk_{idx}")

        self._hfield = self._mjcf_root.asset.add(
            "hfield",
            name=f"terrain_chunk_{idx}_hfield",
            nrow=257,
            ncol=257,
            size=(xyscale, xyscale, zscale, 0.1),
        )
        self._texture = self._mjcf_root.asset.add(
            "texture",
            name=f"terrain_chunk_{idx}_texture",
            width=512,
            height=512,
            builtin="checker",
            rgb1="0.0 0.0 0.5",
            rgb2="0.3 0.3 0.8",
            type="2d",
        )
        self._material = self._mjcf_root.asset.add(
            "material",
            name=f"terrain_chunk_{idx}_material",
            texture=self._texture,
            texuniform="false",
        )
        self._geom = self._mjcf_root.worldbody.add(
            "geom",
            name=f"terrain_chunk_{idx}_geom",
            type="hfield",
            size=(xyscale, xyscale, zscale),
            hfield=f"terrain_chunk_{idx}_hfield",
            material=self._material,
        )
        self._xyscale = xyscale
        self._zscale = zscale
        self._chunk_pos = None
        self._chunk_data = None

    def _fill_chunk(self, physics: Physics, key: ChunkKey, chunk_cache: Dict[ChunkKey, ChunkData]):
        """
        Fill the chunk with terrain data and update physics/rendering engines
        """

        chunk_pos = key.pos

        self._chunk_pos = chunk_pos
        hfield = physics.bind(self._hfield)
        tex = physics.bind(self._texture)

        assert hfield is not None
        assert tex is not None

        # Get heightfield resolution, assert that it is square.
        res = int(hfield.nrow)
        assert res == hfield.ncol

        terrain_res = int(tex.width)
        assert terrain_res == tex.height

        if key not in chunk_cache:
            chunk_cache[key] = _gen_chunk_terrain(key, res, terrain_res)

        chunk_data = chunk_cache[key]
        texture = chunk_data.texture
        terrain = chunk_data.terrain
        self._chunk_data = chunk_data

        physics.model.hfield_data[
            hfield.adr : hfield.adr + res**2
        ] = terrain.transpose(1, 0).ravel()
        if texture is not None:
            physics.model.tex_rgb[tex.adr : tex.adr + texture.size] = texture.transpose(
                1, 0, 2
            ).ravel()

        # Set position for the corresponding geom
        self.set_pose(
            physics,
            (2 * chunk_pos[0] * self._xyscale, 2 * chunk_pos[1] * self._xyscale, 0),
        )

        # If we have a rendering context, we need to re-upload the modified heightfield data.
        if physics.contexts:
            with physics.contexts.gl.make_current() as ctx:
                ctx.call(
                    mjr_upload_hfield,
                    physics.model.ptr,
                    physics.contexts.mujoco.ptr,
                    hfield.element_id,
                )
                if texture is not None:
                    ctx.call(
                        mjr_upload_texture,
                        physics.model.ptr,
                        physics.contexts.mujoco.ptr,
                        tex.element_id,
                    )
        else:
            raise NotImplementedError

    def _lookup_texture_coord(self, pos: np.ndarray):
        assert self._chunk_pos is not None

        # Compute the position in the chunk, in range [0, 1]
        return (
            pos[0] / (2 * self._xyscale) + 0.5 - self._chunk_pos[0],
            pos[1] / (2 * self._xyscale) + 0.5 - self._chunk_pos[1],
        )

    def _texture_coord_to_position(self, coord: np.ndarray) -> np.ndarray:
        assert self._chunk_pos is not None

        return np.asarray([
            (coord[0] - 0.5 + self._chunk_pos[0]) * (2 * self._xyscale),
            (coord[1] - 0.5 + self._chunk_pos[1]) * (2 * self._xyscale),
        ])

    def _lookup_z(self, pos):
        assert self._chunk_data is not None

        # Compute the position in the chunk, in range [0, 1]
        pos_in_chunk = self._lookup_texture_coord(pos)
        idx_i = int(pos_in_chunk[0] * (self._hfield.nrow - 1))
        idx_j = int(pos_in_chunk[1] * (self._hfield.ncol - 1))
        return self._chunk_data.terrain[idx_i, idx_j] * self._zscale

    def _find_nearby_reset_position(
        self, position: np.ndarray, tolerance: float, ntries: int = 10, std: float = 1.5
    ) -> Optional[np.ndarray]:
        assert self._chunk_data is not None

        pos_in_chunk = self._lookup_texture_coord(position)
        mountain = self._chunk_data.aux["mountain"]
        idx_i = int(pos_in_chunk[0] * (mountain.shape[0] - 1))
        idx_j = int(pos_in_chunk[1] * (mountain.shape[1] - 1))

        # Find the nearest reset position
        for _ in range(ntries):
            std_pixels = int(mountain.shape[0] * std / self._xyscale / 2)
            lookup_i = np.random.randint(
                max(0, idx_i - std_pixels),
                min(mountain.shape[0] - 1, idx_i + std_pixels) + 1,
            )
            lookup_j = np.random.randint(
                max(0, idx_j - std_pixels),
                min(mountain.shape[1] - 1, idx_j + std_pixels) + 1,
            )
            if mountain[lookup_i, lookup_j] < tolerance:
                # print(f'Found reset position at {lookup_i}, {lookup_j} from start {idx_i}, {idx_j}')
                pos = self._texture_coord_to_position(
                    np.asarray([
                        lookup_i / (mountain.shape[0] - 1),
                        lookup_j / (mountain.shape[1] - 1),
                    ])
                )
                return np.asarray([*pos, self._lookup_z(pos) + 0.5])

    @property
    def geom(self):
        """The geom belonging to this prop."""
        return self._geom

    @property
    def mjcf_model(self):
        return self._mjcf_root
