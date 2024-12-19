# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.99'

import jax

# jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)

import braintools
import brainunit as u
import diffrax
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import matplotlib.pyplot as plt
import numpy as np
import time
import warnings
from functools import partial
from jaxtyping import Array, Float  # https://github.com/google/jaxtyping
from typing import Callable


# Represents the interval [x0, x_final] discretised into n equally-spaced points.
class SpatialDiscretisation(eqx.Module):
    x0: float = eqx.field(static=True)
    x_final: float = eqx.field(static=True)
    vals: Float[Array, "n"]

    @classmethod
    def discretise_fn(cls, x0: float, x_final: float, n: int, fn: Callable):
        if n < 2:
            raise ValueError("Must discretise [x0, x_final] into at least two points")
        vals = jax.vmap(fn)(u.math.linspace(x0, x_final, n))
        return cls(x0, x_final, vals)

    @property
    def δx(self):
        return (self.x_final - self.x0) / (len(self.vals) - 1)

    def binop(self, other, fn):
        if isinstance(other, SpatialDiscretisation):
            if self.x0 != other.x0 or self.x_final != other.x_final:
                raise ValueError("Mismatched spatial discretisations")
            other = other.vals
        return SpatialDiscretisation(self.x0, self.x_final, fn(self.vals, other))

    def __add__(self, other):
        return self.binop(other, lambda x, y: x + y)

    def __mul__(self, other):
        return self.binop(other, lambda x, y: x * y)

    def __radd__(self, other):
        return self.binop(other, lambda x, y: y + x)

    def __rmul__(self, other):
        return self.binop(other, lambda x, y: y * x)

    def __sub__(self, other):
        return self.binop(other, lambda x, y: x - y)

    def __rsub__(self, other):
        return self.binop(other, lambda x, y: y - x)

    def __truediv__(self, other):
        return self.binop(other, lambda x, y: x / y)


# Problem

@partial(jax.jit, static_argnums=(0, 1))
def _integrate_with_unit(n=21, adaptive: bool = True):
    mu = 1 * u.kilogram / (u.meter * u.second)
    rho = 1 * u.kilogram / u.meter3

    def laplacian(y: SpatialDiscretisation) -> SpatialDiscretisation:
        y_next = u.math.roll(y.vals, shift=1)
        y_prev = u.math.roll(y.vals, shift=-1)

        Δy = (y_next - 2 * y.vals + y_prev) / (y.δx ** 2)

        # Dirichlet boundary condition
        Δy = Δy.at[0].set(0 / (u.meter * u.second))
        Δy = Δy.at[-1].set(0 / (u.meter * u.second))
        return SpatialDiscretisation(y.x0, y.x_final, Δy)

    def vector_field(t, y, args):
        return laplacian(y) * 2e-4 * mu / rho

    term = diffrax.ODETerm(vector_field)

    def ic(x):
        return u.math.where(x == 1 * u.meter, 1.0 * u.meter / u.second, 0.0 * u.meter / u.second)

    # Spatial discretisation
    x0 = 0 * u.meter
    x_final = 1 * u.meter
    y0 = SpatialDiscretisation.discretise_fn(x0, x_final, n, ic)

    # Temporal discretisation
    t0 = 0 * u.second
    t_final = 2500 * u.second
    δt = 1.0 * u.second
    saveat = diffrax.SaveAt(ts=u.math.linspace(t0, t_final, 21))

    # Tolerances
    if adaptive:
        rtol = 1e-10
        atol = 1e-10
        controller = diffrax.PIDController(
            pcoeff=0.3, icoeff=0.4, rtol=rtol, atol=atol  # , dtmax=12.5 * u.second
        )
    else:
        controller = diffrax.ConstantStepSize()

    solver = diffrax.Tsit5()
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0,
        t_final,
        δt,
        y0,
        saveat=saveat,
        stepsize_controller=controller,
        max_steps=None,
        throw=False
    )
    r = sol, x0, x_final, t0, t_final
    return jax.tree.map(lambda x: u.get_mantissa(x), r, is_leaf=u.math.is_quantity)


def _run_simulation(results, with_unit=True):
    sol, x0, x_final, t0, t_final = results

    with_unit = 'with' if with_unit else 'without'

    indices = np.arange(0, sol.ys.vals.shape[0], 2)
    vector = np.linspace(0, 1, 21)

    fig, gs = braintools.visualize.get_figure(1, 1, 5., 5.)
    ax = fig.add_subplot(gs[0, 0])
    for idx in indices:
        plt.plot(sol.ys.vals[idx, :], vector, label=f"{idx * 10}Δt")
    plt.plot(vector, vector, label="ground truth", linestyle='--', color='k')
    plt.xlabel("u")
    plt.ylabel("y")
    plt.legend()
    plt.title(f'Couette Equation Ground Truth Comparison\n(simulation {with_unit} units)')
    plt.savefig(f'results/{with_unit}-unit-ground-truth-compare.svg', transparent=True)

    # plt.figure(figsize=(5, 5))
    fig, gs = braintools.visualize.get_figure(1, 1, 5., 5.)
    ax = fig.add_subplot(gs[0, 0])
    im = plt.imshow(
        sol.ys.vals.T,
        origin="lower",
        extent=(t0,
                t_final,
                x0,
                x_final),
        aspect=(t_final - t0) / (x_final - x0),
        cmap="inferno",
    )
    plt.xlabel("t(s)")
    plt.ylabel("y(m)", rotation=0)
    plt.clim(0, 1)
    plt.colorbar(im, ax=ax)
    plt.title(f'Couette Equation 2D Heatmap\n(simulation {with_unit} units)')
    plt.savefig(f'results/{with_unit}-unit-2d-result.svg', transparent=True)

    ys_np = sol.ys.vals.T
    t_vals = np.linspace(t0, t_final, ys_np.shape[1])
    x_vals = np.linspace(x0, x_final, ys_np.shape[0])
    T, X = np.meshgrid(t_vals, x_vals)

    fig, gs = braintools.visualize.get_figure(1, 1, 5., 5.)
    ax = fig.add_subplot(gs[0, 0], projection='3d')
    surf = ax.plot_surface(T, X, ys_np, cmap='inferno', edgecolor='none', vmin=0., vmax=1.)
    ax.set_xlabel('t (s)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('u(m/s)')
    fig.colorbar(surf)
    ax.view_init(elev=45, azim=235)
    plt.title(f'Couette Equation 3D Surface Plot\n(simulation {with_unit} units)')
    plt.savefig(f'results/{with_unit}-unit-3d-result.svg', transparent=True)

    plt.show()


def _run_simulation_with_unit():
    sol, x0, x_final, t0, t_final = _integrate_with_unit()
    _run_simulation([sol, x0, x_final, t0, t_final], True)


# Problem

@partial(jax.jit, static_argnums=(0, 1))
def _integrate_without_unit(n=21, adaptive: bool = True):
    mu = 1
    rho = 1

    def laplacian(y: SpatialDiscretisation) -> SpatialDiscretisation:
        y_next = u.math.roll(y.vals, shift=1)
        y_prev = u.math.roll(y.vals, shift=-1)

        Δy = (y_next - 2 * y.vals + y_prev) / (y.δx ** 2)

        # Dirichlet boundary condition
        Δy = Δy.at[0].set(0)
        Δy = Δy.at[-1].set(0)
        return SpatialDiscretisation(y.x0, y.x_final, Δy)

    def vector_field(t, y, args):
        return laplacian(y) * 2e-4 * mu / rho

    term = diffrax.ODETerm(vector_field)

    def ic(x):
        return u.math.where(x == 1, 1.0, 0.0)

    # Spatial discretisation
    x0 = 0
    x_final = 1
    y0 = SpatialDiscretisation.discretise_fn(x0, x_final, n, ic)

    # Temporal discretisation
    t0 = 0
    t_final = 2500
    δt = 1.0
    saveat = diffrax.SaveAt(ts=u.math.linspace(t0, t_final, 21))

    # Tolerances
    if adaptive:
        rtol = 1e-10
        atol = 1e-10
        controller = diffrax.PIDController(
            pcoeff=0.3, icoeff=0.4, rtol=rtol, atol=atol  # , dtmax=12.5 * u.second
        )
    else:
        controller = diffrax.ConstantStepSize()

    solver = diffrax.Tsit5()
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0,
        t_final,
        δt,
        y0,
        saveat=saveat,
        stepsize_controller=controller,
        max_steps=None,
        throw=False
    )
    r = sol, x0, x_final, t0, t_final
    return jax.tree.map(lambda x: u.get_mantissa(x), r, is_leaf=u.math.is_quantity)


def _run_simulation_without_unit():
    sol, x0, x_final, t0, t_final = _integrate_without_unit()
    _run_simulation([sol, x0, x_final, t0, t_final], False)


def compare_simulation_results():
    _run_simulation_with_unit()
    _run_simulation_without_unit()


def compare_compilation_time():
    import statistics
    from scipy import stats
    import pandas as pd

    warnings.filterwarnings("ignore", category=UserWarning)
    platform = jax.default_backend()

    n_run = 10

    heads = ['n', 'unit_or_not', 'time']
    results = []

    for n in [10, 100, 1000, 10000, 100000]:
        with_unit_compile_times = []
        for _ in range(n_run):
            t0 = time.time()
            _integrate_with_unit.lower(n).compile()
            t1 = time.time()
            with_unit_compile_times.append(t1 - t0)

            jax.clear_caches()
            eqx.clear_caches()

            results.append([n, 'with unit', t1 - t0])

        with_unit_mean = statistics.mean(with_unit_compile_times)
        with_unit_std = statistics.stdev(with_unit_compile_times)
        print(f'with unit compile time = {with_unit_mean} ± {with_unit_std} s')

        without_unit_compile_times = []
        for _ in range(n_run):
            t0 = time.time()
            _integrate_without_unit.lower(n).compile()
            t1 = time.time()
            without_unit_compile_times.append(t1 - t0)

            jax.clear_caches()
            eqx.clear_caches()

            results.append([n, 'without unit', t1 - t0])

        without_unit_mean = statistics.mean(without_unit_compile_times)
        without_unit_std = statistics.stdev(without_unit_compile_times)
        print(f'with unit compile time = {without_unit_mean} ± {without_unit_std} s')

        tstat, pval = stats.ttest_ind(a=with_unit_compile_times, b=without_unit_compile_times, alternative="two-sided")
        print("t-stat: {:.2f}   pval: {:.4f}".format(tstat, pval))
        print()

    pd.DataFrame(results, columns=heads).to_csv(f'results/couette-{platform}-compile-time.csv', index=False)


def compare_simulation_time():
    import statistics
    from scipy import stats
    import pandas as pd

    warnings.filterwarnings("ignore", category=UserWarning)

    platform = jax.default_backend()

    heads = ['n', 'unit_or_not', 'time']
    results = []
    n_run = [100, 100, 10, 10, 5]
    n_run = [100, 100, 100, 100, 100]

    for i, n in enumerate([10, 100, 1000, 10000, 100000]):

        jax.block_until_ready(_integrate_with_unit(n, False))  # compile the model with units
        with_unit_simulate_times = []
        for _ in range(n_run[i]):
            t0 = time.time()
            jax.block_until_ready(_integrate_with_unit(n, False))
            t1 = time.time()
            with_unit_simulate_times.append(t1 - t0)

            results.append([n, 'with unit', t1 - t0])

        print(f'with unit simulation time = {statistics.mean(with_unit_simulate_times)} '
              f'± {statistics.stdev(with_unit_simulate_times)} s')

        without_unit_simulate_times = []
        jax.block_until_ready(_integrate_without_unit(n, False))  # compile the model without units
        for _ in range(n_run[i]):
            t0 = time.time()
            jax.block_until_ready(_integrate_without_unit(n, False))
            t1 = time.time()
            without_unit_simulate_times.append(t1 - t0)

            results.append([n, 'without unit', t1 - t0])

        print(f'without unit simulation time = {statistics.mean(without_unit_simulate_times)} '
              f'± {statistics.stdev(without_unit_simulate_times)} s')

        tstat, pval = stats.ttest_ind(a=with_unit_simulate_times,
                                      b=without_unit_simulate_times,
                                      alternative="two-sided")
        print("t-stat: {:.2f}   pval: {:.4f}".format(tstat, pval))
        print()

        jax.clear_caches()
        eqx.clear_caches()

    pd.DataFrame(results, columns=heads).to_csv(f'results/couette-{platform}-simulation-time.csv', index=False)


if __name__ == '__main__':
    pass

    # compare_simulation_results()
    # compare_compilation_time()
    compare_simulation_time()
