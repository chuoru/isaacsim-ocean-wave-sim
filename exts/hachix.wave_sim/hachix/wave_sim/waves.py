"""Trochoidal (Gerstner) wavefield, following wave_sim_vrx `Wavefield.cc` (Apache-2.0).

Convention, per component i with theta_i = k_i (d_i . x0) - omega_i t + phi_i:
    x = x0 - sum_i q_i a_i d_i sin(theta_i)
    z =      sum_i a_i cos(theta_i)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

GRAVITY = 9.81

NEWTON_TOLERANCE = 1.0e-10
NEWTON_MAX_ITERATIONS = 30


def dispersion_omega(wavenumber: np.ndarray | float) -> np.ndarray | float:
    """Deep water dispersion omega^2 = g k."""
    return np.sqrt(GRAVITY * wavenumber)


def dispersion_wavenumber(omega: np.ndarray | float) -> np.ndarray | float:
    return omega * omega / GRAVITY


@dataclass(frozen=True)
class WaveParameters:
    """Same fields and defaults as the VRX `<wave>` SDF element."""

    number: int = 1
    scale: float = 2.0
    angle: float = 2.0 * np.pi / 10.0
    steepness: float = 1.0
    amplitude: float = 0.0
    period: float = 1.0
    phase: float = 0.0
    direction: tuple[float, float] = (1.0, 0.0)

    def __post_init__(self) -> None:
        if self.number < 1:
            raise ValueError("number must be >= 1")
        if self.period <= 0.0:
            raise ValueError("period must be > 0")
        if self.amplitude < 0.0:
            raise ValueError("amplitude must be >= 0")
        if np.hypot(*self.direction) == 0.0:
            raise ValueError("direction must be non-zero")

    @classmethod
    def vrx_ocean_waves(cls, **overrides) -> "WaveParameters":
        """Defaults of `wave_gazebo/world_models/ocean_waves/model.sdf`."""
        values = dict(number=3, scale=2.5, angle=0.3, steepness=1.0, amplitude=0.0, period=8.0, direction=(1.0, 1.0))
        values.update(overrides)
        return cls(**values)


@dataclass(frozen=True)
class WaveComponents:
    amplitude: np.ndarray
    wavenumber: np.ndarray
    omega: np.ndarray
    phase: np.ndarray
    steepness: np.ndarray
    direction: np.ndarray = field(repr=False)


def wave_components(params: WaveParameters) -> WaveComponents:
    """Mirror of VRX `WaveParametersPrivate::Recalculate`."""
    n = np.arange(params.number) - params.number // 2
    scale_factor = params.scale ** n.astype(float)
    mean_k = dispersion_wavenumber(2.0 * np.pi / params.period)
    a = scale_factor * params.amplitude
    k = mean_k / scale_factor
    with np.errstate(divide="ignore"):
        q = np.where(a != 0.0, np.minimum(1.0, params.steepness / (a * k * params.number)), 0.0)
    d0 = np.asarray(params.direction, dtype=float) / np.hypot(*params.direction)
    c, s = np.cos(n * params.angle), np.sin(n * params.angle)
    direction = np.stack([c * d0[0] - s * d0[1], s * d0[0] + c * d0[1]], axis=1)
    return WaveComponents(
        amplitude=a,
        wavenumber=k,
        omega=dispersion_omega(k),
        phase=np.full(params.number, params.phase),
        steepness=q,
        direction=direction,
    )


class Wavefield:
    def __init__(self, components: WaveComponents) -> None:
        self.components = components
        c = components
        self._qa_d = (c.steepness * c.amplitude)[:, None] * c.direction
        self._k_d = c.wavenumber[:, None] * c.direction
        self._qak = c.steepness * c.amplitude * c.wavenumber
        self._dd = c.direction[:, :, None] * c.direction[:, None, :]

    @classmethod
    def from_parameters(cls, params: WaveParameters) -> "Wavefield":
        return cls(wave_components(params))

    def _theta(self, xy: np.ndarray, t: float) -> np.ndarray:
        return xy @ self._k_d.T - self.components.omega * t + self.components.phase

    def displace(self, grid_xy: np.ndarray, t: float) -> np.ndarray:
        """Gerstner forward map of rest positions [M,2] to surface points [M,3]."""
        theta = self._theta(grid_xy, t)
        out = np.empty((grid_xy.shape[0], 3))
        out[:, :2] = grid_xy - np.sin(theta) @ self._qa_d
        out[:, 2] = np.cos(theta) @ self.components.amplitude
        return out

    def height(self, xy: np.ndarray, t: float) -> np.ndarray:
        """Surface height at fixed world xy [M,2], by Newton on the rest position (VRX ComputeDepthDirectly).

        VRX's residual omits the steepness q while its Jacobian keeps it, so it converges to a
        point that is not on the rendered surface whenever q != 1. Here both use q consistently.
        """
        x0 = np.array(xy, dtype=float, copy=True)
        active = np.ones(x0.shape[0], dtype=bool)
        for _ in range(NEWTON_MAX_ITERATIONS):
            idx = np.flatnonzero(active)
            if idx.size == 0:
                break
            theta = self._theta(x0[idx], t)
            residual = x0[idx] - np.sin(theta) @ self._qa_d - xy[idx]
            jac = np.eye(2) - np.einsum("mi,i,ijk->mjk", np.cos(theta), self._qak, self._dd)
            x0[idx] -= np.linalg.solve(jac, residual[:, :, None])[:, :, 0]
            active[idx] = np.linalg.norm(residual, axis=1) > NEWTON_TOLERANCE
        return np.cos(self._theta(x0, t)) @ self.components.amplitude
