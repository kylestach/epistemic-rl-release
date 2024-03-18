from dm_control import composer
import numpy as np
from scipy import ndimage
from procedural_driving import chunk
from procedural_driving.procedural import procedural_random, procedural_generation
import itertools
import os
from typing import Dict

CHUNK_SCALE = 10
CHUNK_SIZE = 2 * CHUNK_SCALE

chunk_cache: Dict[chunk.ChunkKey, chunk.ChunkData] = {}


class ProceduralArena(composer.Arena):
    """An infinite procedurally-generated arena."""

    def _build(self, world_name="default", seed=0):
        """
        Build the arena, and create the chunks, but don't give them any actual heightmap or texture data yet.
        """
        super()._build(name=world_name)

        self._seed = seed
        procedural_random.set_global_seed(self._seed)
        procedural_generation.set_config_name(world_name)

        # Skybox texture from dm_control's outdoor_natural textures
        texturedir = os.path.join(os.path.dirname(__file__), "textures")
        self._mjcf_root.compiler.texturedir = texturedir

        self._skybox = self._mjcf_root.asset.add(
            "texture",
            name="aesthetic_skybox",
            file="skybox.png",
            type="skybox",
            gridsize="3 4",
            gridlayout=".U..LFRB.D..",
        )

        # Create empty
        self.chunks = []
        for i in range(9):
            self.chunks.append(
                chunk.Chunk(
                    idx=i,
                )
            )
            self.attach(self.chunks[-1])

        self._mjcf_root.visual.headlight.set_attributes(
            ambient=[0.4, 0.4, 0.4], diffuse=[0.8, 0.8, 0.8], specular=[0.1, 0.1, 0.1]
        )

        self.last_chunk = (0, 0)

        global chunk_cache
        self.chunk_cache = chunk_cache
        self.chunk_map = {}

        # Set zfar
        self._mjcf_root.visual.map.set_attributes(znear=0.001, zfar=100)

    def initialize_episode(self, physics, random_state: np.random.RandomState):
        """
        Initialize the episode by generating the terrain and textures for the current chunk and its neighbors.
        """
        self.last_chunk = (0, 0)
        self.chunk_map = {}
        for chunk_idx, chunk_pos in enumerate(
            itertools.product(range(-1, 2), range(-1, 2))
        ):
            chunk_key = chunk.ChunkKey(chunk_pos, self._seed)
            self.chunk_map[chunk_pos] = chunk_idx
            self.chunks[chunk_idx]._fill_chunk(physics, chunk_key, self.chunk_cache)

    def _lookup_chunk(self, pos):
        """
        Lookup the chunk from an xy position
        """
        return (
            int(np.floor((pos[0] + CHUNK_SCALE) / CHUNK_SIZE)),
            int(np.floor((pos[1] + CHUNK_SCALE) / CHUNK_SIZE)),
        )

    def _lookup_z(self, pos):
        """
        Lookup the z value of the terrain at the given position.
        """
        chunk = self._lookup_chunk(pos)
        chunk_idx = self.chunk_map[chunk]
        return self.chunks[chunk_idx]._lookup_z(pos)

    def _set_chunk(self, physics, center_chunk):
        """
        Set the current center chunk (where the robot is) and ensure all surrounding chunks are loaded.
        """

        # Compute the list of chunks that we need.
        def get_chunks(center_chunk):
            return set(
                (center_chunk[0] + i, center_chunk[1] + j)
                for i in range(-1, 2)
                for j in range(-1, 2)
            )

        new_chunks = get_chunks(center_chunk)
        old_chunks = get_chunks(self.last_chunk)

        freed_chunks = old_chunks - new_chunks
        alloc_chunks = new_chunks - old_chunks

        # Figure out which chunks we don't need anymore
        free_geom_idcs = set(self.chunk_map[c] for c in freed_chunks)
        for c in freed_chunks:
            del self.chunk_map[c]
        self.last_chunk = center_chunk

        # Compute new chunks
        assert len(alloc_chunks) <= len(free_geom_idcs)
        for chunk_idx, chunk_pos in zip(free_geom_idcs, alloc_chunks):
            self.chunk_map[chunk_pos] = chunk_idx
            self.chunks[chunk_idx]._fill_chunk(
                physics, chunk.ChunkKey(chunk_pos, self._seed), self.chunk_cache
            )

    def _find_nearby_reset_position(self, position: np.ndarray):
        """
        Find a resettable position nearest to the given position.

        "resettable" is defined as a position that is on a path, rather than a mountain.
        If a suitable position cannot be found, tolerance is increased and the search is repeated.
        """
        # Get a random position in the current chunk
        chunk_loc = self._lookup_chunk(position)
        if chunk_loc not in self.chunk_map:
            return None
        chunk = self.chunks[self.chunk_map[chunk_loc]]

        # Find the nearest flat point in the chunk by rejection sampling
        for k in range(20):
            tolerance = 0.05 * (k + 1)
            pos = chunk._find_nearby_reset_position(position, tolerance=tolerance)
            if pos is not None:
                return pos
            print(
                f"Failed to find a nearby reset position, increasing tolerance to {tolerance}"
            )

    def _find_nearby_goal(self, position: np.ndarray, tolerance: float):
        """
        Find a goal position near the given position.

        Uses the same rejection sampling as _find_nearby_reset_position.
        """
        chunk_loc = self._lookup_chunk(position)
        if chunk_loc not in self.chunk_map:
            return None
        chunk = self.chunks[self.chunk_map[chunk_loc]]
        return chunk._find_nearby_reset_position(position, tolerance=tolerance)
