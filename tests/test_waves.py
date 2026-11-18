import numpy as np
import pytest

from hachix.wave_sim.waves import GRAVITY, Wavefield, WaveParameters, wave_components

SEA_STATES = {
    "vrx_defaults_0.5m_6s": WaveParameters.vrx_ocean_waves(amplitude=0.5, period=6.0),
    "partial_steepness": WaveParameters.vrx_ocean_waves(amplitude=0.8, period=3.0, steepness=0.6),
    "single_component": WaveParameters(number=1, amplitude=0.3, period=4.0, direction=(0.0, 1.0)),
}


@pytest.mark.parametrize("params", SEA_STATES.values(), ids=SEA_STATES.keys())
@pytest.mark.parametrize("t", [0.0, 1.7, 23.4])
def test_height_inverts_forward_map(params, t):
    field = Wavefield.from_parameters(params)
    rest = np.random.default_rng(0).uniform(-40.0, 40.0, size=(2000, 3))[:, :2]
    surface = field.displace(rest, t)
    error = np.abs(field.height(surface[:, :2], t) - surface[:, 2])
    assert error.max() < 1e-6


def test_zero_amplitude_is_flat():
    field = Wavefield.from_parameters(WaveParameters.vrx_ocean_waves(amplitude=0.0))
    xy = np.random.default_rng(1).uniform(-10.0, 10.0, size=(100, 2))
    np.testing.assert_array_equal(field.height(xy, 3.0), 0.0)
    np.testing.assert_array_equal(field.displace(xy, 3.0), np.column_stack([xy, np.zeros(len(xy))]))


def test_components_obey_deep_water_dispersion():
    params = WaveParameters.vrx_ocean_waves(amplitude=0.5, period=6.0)
    c = wave_components(params)
    np.testing.assert_allclose(c.omega**2, GRAVITY * c.wavenumber)
    mean = params.number // 2
    assert 2.0 * np.pi / c.omega[mean] == pytest.approx(params.period)
    np.testing.assert_allclose(c.amplitude, 0.5 * 2.5 ** np.array([-1.0, 0.0, 1.0]))


def test_crest_travels_at_deep_water_phase_speed():
    period = 6.0
    field = Wavefield.from_parameters(WaveParameters(number=1, amplitude=0.4, period=period, direction=(1.0, 0.0)))
    phase_speed = GRAVITY * period / (2.0 * np.pi)
    xy = np.column_stack([np.linspace(-20.0, 20.0, 50), np.zeros(50)])
    dt = 0.8
    later = field.height(xy + [phase_speed * dt, 0.0], 5.0 + dt)
    np.testing.assert_allclose(later, field.height(xy, 5.0), atol=1e-9)


def test_component_slopes_never_fold_the_surface():
    c = wave_components(WaveParameters.vrx_ocean_waves(amplitude=0.5, period=6.0))
    assert (c.steepness * c.amplitude * c.wavenumber).sum() <= 1.0 + 1e-12


def test_partial_steepness_case_exercises_q_below_one():
    assert (wave_components(SEA_STATES["partial_steepness"]).steepness < 1.0).all()
