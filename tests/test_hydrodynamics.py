import numpy as np
import pytest

from hachix.wave_sim.hull import HullMesh
from hachix.wave_sim.hydrodynamics import WATER_DENSITY, HydroParameters, Pose, compute_wrench
from hachix.wave_sim.waves import GRAVITY, Wavefield, WaveParameters

SIZE = (2.0, 1.5, 0.3)
MASS = 150.0
HULL = HullMesh.box(SIZE)
CALM = Wavefield.from_parameters(WaveParameters(amplitude=0.0))
STATIC = HydroParameters(damping_on=False, pressure_drag_on=False)
IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])
ZERO = np.zeros(3)


def wrench_at(z, orientation=IDENTITY, lin_vel=ZERO, ang_vel=ZERO, params=STATIC, wavefield=CALM, t=0.0):
    pose = Pose(np.array([0.0, 0.0, z]), orientation)
    return compute_wrench(HULL, pose, lin_vel, ang_vel, wavefield, t, params)


def roll_quaternion(angle):
    return np.array([np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0])


def test_box_hull_is_closed_with_outward_normals():
    assert HULL.volume == pytest.approx(np.prod(SIZE))


def test_equilibrium_draft_supports_weight():
    draft = MASS / (WATER_DENSITY * SIZE[0] * SIZE[1])
    w = wrench_at(SIZE[2] / 2 - draft)
    assert w.force[2] == pytest.approx(MASS * GRAVITY, rel=0.02)
    np.testing.assert_allclose(w.force[:2], 0.0, atol=1e-9)
    np.testing.assert_allclose(w.torque, 0.0, atol=1e-9)


def test_fully_above_water_has_no_force():
    w = wrench_at(1.0, lin_vel=np.array([0.3, 0.0, -1.0]), params=HydroParameters())
    np.testing.assert_array_equal(w.force, 0.0)
    np.testing.assert_array_equal(w.torque, 0.0)


@pytest.mark.parametrize("orientation", [IDENTITY, roll_quaternion(np.pi / 2), np.array([0.8, 0.2, -0.3, 0.4])])
def test_fully_submerged_buoyancy_is_displaced_weight(orientation):
    w = wrench_at(-3.0, orientation=orientation)
    np.testing.assert_allclose(w.force, [0.0, 0.0, WATER_DENSITY * GRAVITY * np.prod(SIZE)], rtol=1e-9, atol=1e-6)
    np.testing.assert_allclose(w.torque, 0.0, atol=1e-6)


@pytest.mark.parametrize("angle", [0.05, -0.1])
def test_heeled_raft_gets_righting_moment(angle):
    draft = MASS / (WATER_DENSITY * SIZE[0] * SIZE[1])
    w = wrench_at(SIZE[2] / 2 - draft, orientation=roll_quaternion(angle))
    assert np.sign(w.torque[0]) == -np.sign(angle)


def test_drag_and_damping_oppose_heave_velocity():
    z = SIZE[2] / 2 - 0.05
    still = wrench_at(z, params=HydroParameters()).force[2]
    sinking = wrench_at(z, lin_vel=np.array([0.0, 0.0, -0.5]), params=HydroParameters()).force[2]
    rising = wrench_at(z, lin_vel=np.array([0.0, 0.0, 0.5]), params=HydroParameters()).force[2]
    assert sinking > still > rising


def test_wave_crest_lifts_more_than_trough():
    field = Wavefield.from_parameters(WaveParameters(number=1, amplitude=0.3, period=8.0))
    period = 8.0
    crest = wrench_at(0.0, wavefield=field, t=0.0).force[2]
    trough = wrench_at(0.0, wavefield=field, t=period / 2).force[2]
    assert crest > trough
