"""Hull hydrodynamics on a Gerstner wavefield.

Method as in asv_wave_sim `Hydrodynamics` (after Kerner, "Water interaction model for boats in
video games"): clip hull triangles at the water surface, integrate hydrostatic pressure over the
submerged parts, add pressure drag and linear/quadratic damping.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hull import HullMesh
from .waves import GRAVITY, Wavefield

WATER_DENSITY = 1025.0


@dataclass(frozen=True)
class HydroParameters:
    """asv_wave_sim `HydrodynamicsParameters`, minus viscous drag (not ported)."""

    damping_on: bool = True
    pressure_drag_on: bool = True
    cDampL1: float = 1.0e-6
    cDampL2: float = 1.0e-6
    cDampR1: float = 1.0e-6
    cDampR2: float = 1.0e-6
    cPDrag1: float = 1.0e2
    cPDrag2: float = 1.0e2
    fPDrag: float = 0.4
    cSDrag1: float = 1.0e2
    cSDrag2: float = 1.0e2
    fSDrag: float = 0.4
    vRDrag: float = 1.0
    water_density: float = WATER_DENSITY


@dataclass(frozen=True)
class Pose:
    position: np.ndarray
    orientation_wxyz: np.ndarray

    def rotation(self) -> np.ndarray:
        w, x, y, z = np.asarray(self.orientation_wxyz, dtype=float) / np.linalg.norm(self.orientation_wxyz)
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
        )


@dataclass(frozen=True)
class Wrench:
    force: np.ndarray
    torque: np.ndarray


# Per triangle, vertices sorted by depth ascending (shallowest first) give six candidate points:
#   0 H, 1 M, 2 L (deepest), 3 X(L,M), 4 X(L,H), 5 X(M,H)   where X(a,b) is the waterline crossing on edge a-b.
# Indexed by the number of submerged vertices, each row lists up to two sub-triangles covering the wet part.
_CLIP_POINTS = np.array(
    [
        [[0, 0, 0], [0, 0, 0]],
        [[2, 3, 4], [0, 0, 0]],
        [[1, 5, 2], [5, 4, 2]],
        [[0, 1, 2], [0, 0, 0]],
    ]
)
_CLIP_VALID = np.array([[False, False], [True, False], [True, True], [True, False]])
_EDGES = np.array([[2, 1], [2, 0], [1, 0]])


def _crossings(points: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Waterline crossing on each edge in `_EDGES`, by linear interpolation of depth. [T,3,3]."""
    a, b = _EDGES[:, 0], _EDGES[:, 1]
    da, db = depth[:, a], depth[:, b]
    denom = da - db
    s = np.divide(da, denom, out=np.zeros_like(da), where=denom != 0.0)
    return points[:, a] + (points[:, b] - points[:, a]) * s[..., None]


def submerged_triangles(
    world_vertices: np.ndarray, triangles: np.ndarray, vertex_depth: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clip every triangle at the waterline.

    Returns (parent [S], corners [S,3,3], corner_depth [S,3]) for the S wet sub-triangles.
    Depth is positive below the surface and treated as linear across each hull triangle.
    """
    depth = vertex_depth[triangles]
    order = np.argsort(depth, axis=1)
    depth = np.take_along_axis(depth, order, axis=1)
    points = np.take_along_axis(world_vertices[triangles], order[..., None], axis=1)

    candidates = np.concatenate([points, _crossings(points, depth)], axis=1)
    candidate_depth = np.concatenate([depth, np.zeros_like(depth)], axis=1)

    wet_count = (depth > 0.0).sum(axis=1)
    parent, slot = np.nonzero(_CLIP_VALID[wet_count])
    corner_idx = _CLIP_POINTS[wet_count[parent], slot]
    return parent, candidates[parent[:, None], corner_idx], candidate_depth[parent[:, None], corner_idx]


def center_of_pressure(corners: np.ndarray, corner_depth: np.ndarray) -> np.ndarray:
    """Exact pressure centroid of a triangle under linearly varying pressure, from integrals of barycentric products."""
    total = corner_depth.sum(axis=1)
    weighted = np.einsum("si,sij->sj", corner_depth, corners) + total[:, None] * corners.sum(axis=1)
    centroid = corners.mean(axis=1)
    return np.divide(weighted, 4.0 * total[:, None], out=centroid, where=total[:, None] > 0.0)


def _pressure_drag(normal, point_velocity, area, params: HydroParameters) -> np.ndarray:
    speed = np.linalg.norm(point_velocity, axis=1)
    unit = np.divide(point_velocity, speed[:, None], out=np.zeros_like(point_velocity), where=speed[:, None] > 0.0)
    cos_theta = np.einsum("ij,ij->i", unit, normal)
    v = speed / params.vRDrag
    push = -(params.cPDrag1 * v + params.cPDrag2 * v * v) * area * np.abs(cos_theta) ** params.fPDrag
    suction = (params.cSDrag1 * v + params.cSDrag2 * v * v) * area * np.abs(cos_theta) ** params.fSDrag
    return normal * np.where(cos_theta >= 0.0, push, suction)[:, None]


def compute_wrench(
    hull: HullMesh,
    pose: Pose,
    lin_vel: np.ndarray,
    ang_vel: np.ndarray,
    wavefield: Wavefield,
    t: float,
    params: HydroParameters,
) -> Wrench:
    """Hydrodynamic force and torque about the center of mass, world frame. Gravity is not included."""
    rotation = pose.rotation()
    com = np.asarray(pose.position, dtype=float)
    world = hull.vertices @ rotation.T + com
    vertex_depth = wavefield.height(world[:, :2], t) - world[:, 2]

    tri = world[hull.triangles]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    tri_area = 0.5 * np.linalg.norm(cross, axis=1)
    tri_normal = cross / (2.0 * tri_area[:, None])

    parent, corners, corner_depth = submerged_triangles(world, hull.triangles, vertex_depth)
    normal = tri_normal[parent]
    area = 0.5 * np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1)

    rho_g = params.water_density * GRAVITY
    buoyancy = -(rho_g * corner_depth.mean(axis=1) * area)[:, None] * normal
    force = buoyancy.sum(axis=0)
    torque = np.cross(center_of_pressure(corners, corner_depth) - com, buoyancy).sum(axis=0)

    if params.pressure_drag_on:
        arm = corners.mean(axis=1) - com
        drag = _pressure_drag(normal, lin_vel + np.cross(ang_vel, arm), area, params)
        force = force + drag.sum(axis=0)
        torque = torque + np.cross(arm, drag).sum(axis=0)

    if params.damping_on:
        wet_ratio = area.sum() / tri_area.sum()
        lin_speed = np.linalg.norm(lin_vel)
        ang_speed = np.linalg.norm(ang_vel)
        force = force - wet_ratio * (params.cDampL1 + params.cDampL2 * lin_speed) * np.asarray(lin_vel)
        torque = torque - wet_ratio * (params.cDampR1 + params.cDampR2 * ang_speed) * np.asarray(ang_vel)

    return Wrench(force=force, torque=torque)
