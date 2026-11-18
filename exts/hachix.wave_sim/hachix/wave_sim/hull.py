from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HullMesh:
    """Closed triangle mesh in the body frame, origin at the center of mass, outward normals."""

    vertices: np.ndarray
    triangles: np.ndarray

    @property
    def volume(self) -> float:
        v = self.vertices[self.triangles]
        return float(np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])).sum() / 6.0)

    @classmethod
    def from_polygons(cls, points: np.ndarray, face_counts: np.ndarray, face_indices: np.ndarray) -> "HullMesh":
        """Fan-triangulate USD-style polygons (convex faces assumed), keeping their winding."""
        counts = np.asarray(face_counts, dtype=np.int64)
        indices = np.asarray(face_indices, dtype=np.int64)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        fan = [
            np.stack([np.full(n - 2, indices[s]), indices[s + 1 : s + n - 1], indices[s + 2 : s + n]], axis=1)
            for s, n in zip(starts, counts)
            if n >= 3
        ]
        return cls(np.asarray(points, dtype=float), np.concatenate(fan))

    @classmethod
    def box(cls, size: tuple[float, float, float], cells_per_meter: float = 8.0) -> "HullMesh":
        """Axis-aligned box centered at the origin, each face split into a grid of triangle pairs."""
        half = np.asarray(size, dtype=float) / 2.0
        vertices: list[np.ndarray] = []
        triangles: list[np.ndarray] = []
        offset = 0
        for axis in range(3):
            u_axis, v_axis = (axis + 1) % 3, (axis + 2) % 3
            nu = max(1, int(np.ceil(size[u_axis] * cells_per_meter)))
            nv = max(1, int(np.ceil(size[v_axis] * cells_per_meter)))
            u = np.linspace(-half[u_axis], half[u_axis], nu + 1)
            v = np.linspace(-half[v_axis], half[v_axis], nv + 1)
            uu, vv = np.meshgrid(u, v, indexing="ij")
            ij = np.arange((nu + 1) * (nv + 1)).reshape(nu + 1, nv + 1)
            a, b, c, d = ij[:-1, :-1].ravel(), ij[1:, :-1].ravel(), ij[1:, 1:].ravel(), ij[:-1, 1:].ravel()
            # (u_axis, v_axis, axis) is a right-handed cycle, so (a, b, c) winds toward +axis.
            quads = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
            for sign in (1.0, -1.0):
                pts = np.zeros((uu.size, 3))
                pts[:, u_axis] = uu.ravel()
                pts[:, v_axis] = vv.ravel()
                pts[:, axis] = sign * half[axis]
                tris = quads if sign > 0 else quads[:, ::-1]
                vertices.append(pts)
                triangles.append(tris + offset)
                offset += pts.shape[0]
        return cls(np.concatenate(vertices), np.concatenate(triangles).astype(np.int64))
