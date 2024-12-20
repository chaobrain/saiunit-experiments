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


import time
from functools import partial

import brainunit as u
import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

unit_of_x = u.meter
unit_of_y = u.meter
unit_of_t = u.second
unit_of_speed = u.meter / u.second
unit_of_psi = u.meter ** 2 / u.second
unit_of_f = u.meter ** 2 / u.second ** 2


@partial(jax.jit, static_argnums=(0, 1))
def _integrate_with_unit(n=64, adaptive: bool = True):
    Nx, Ny = n, n

    # 定义离散化操作
    def laplacian(phi, dx, dy):
        return (
            (u.math.roll(phi, -1, axis=0) - 2 * phi + u.math.roll(phi, 1, axis=0)) / dx ** 2 +
            (u.math.roll(phi, -1, axis=1) - 2 * phi + u.math.roll(phi, 1, axis=1)) / dy ** 2
        )

    def solve_poisson(omega, dx, dy):
        omega = omega / unit_of_speed
        dx = dx / unit_of_x
        dy = dy / unit_of_y
        Nx, Ny = omega.shape
        kx = jnp.fft.fftfreq(Nx, d=dx) * 2 * jnp.pi
        ky = jnp.fft.fftfreq(Ny, d=dy) * 2 * jnp.pi
        Kx, Ky = jnp.meshgrid(kx, ky, indexing='ij')
        K_squared = Kx ** 2 + Ky ** 2
        K_squared = jnp.where(K_squared == 0, 1.0, K_squared)  # 避免除以零
        omega_hat = jnp.fft.fft2(omega)
        psi_hat = -omega_hat / K_squared
        psi = jnp.real(jnp.fft.ifft2(psi_hat))
        return psi

    def navier_stokes_ode(t, state, args):
        omega, psi = state
        dx, dy, nu, f = args

        # 计算速度场
        u_ = (u.math.roll(psi, -1, axis=1) - u.math.roll(psi, 1, axis=1)) / (2 * dy)
        v = -(u.math.roll(psi, -1, axis=0) - u.math.roll(psi, 1, axis=0)) / (2 * dx)

        # 计算涡度的对流项
        domega_dx = (u.math.roll(omega, -1, axis=0) - u.math.roll(omega, 1, axis=0)) / (2 * dx)
        domega_dy = (u.math.roll(omega, -1, axis=1) - u.math.roll(omega, 1, axis=1)) / (2 * dy)
        advect = u_ * domega_dx + v * domega_dy

        # 计算涡度的扩散项
        diffusion = nu * laplacian(omega, dx, dy)

        # 计算外力的涡度贡献
        f_vorticity = (
            (u.math.roll(f[1], -1, axis=0) - u.math.roll(f[1], 1, axis=0)) / (2 * dx) -
            (u.math.roll(f[0], -1, axis=1) - u.math.roll(f[0], 1, axis=1)) / (2 * dy)
        )

        # d(omega)/dt
        domega_dt = -advect + diffusion + f_vorticity

        # 由于 psi 是通过泊松方程即时更新的，dpsi_dt 为零
        dpsi_dt = jnp.zeros_like(psi) * unit_of_psi / u.second

        return (domega_dt, dpsi_dt)

    def initial_conditions(Nx, Ny, dx, dy):
        # 示例：初始化为一个中心涡旋
        x = jnp.linspace(0, 2 * jnp.pi, Nx, endpoint=False)
        y = jnp.linspace(0, 2 * jnp.pi, Ny, endpoint=False)
        X, Y = jnp.meshgrid(x, y, indexing='ij')
        omega0 = jnp.sin(X) * jnp.sin(Y) * unit_of_speed
        psi0 = solve_poisson(omega0, dx, dy)
        return omega0, psi0 * unit_of_psi

    # 设置网格和物理参数
    Lx, Ly = 2.0 * jnp.pi * unit_of_x, 2.0 * jnp.pi * unit_of_y
    dx, dy = Lx / Nx, Ly / Ny
    nu = 1e-3 * u.meter ** 2 / u.second  # 粘性系数
    f = (jnp.zeros((Nx, Ny)) * unit_of_f, jnp.zeros((Nx, Ny)) * unit_of_f)  # 无外力

    # 初始条件
    state0 = initial_conditions(Nx, Ny, dx, dy)

    # 时间范围
    t0 = 0.0 * unit_of_t
    t1 = 10.0 * unit_of_t

    # ODE 参数
    args = (dx, dy, nu, f)

    # 创建求解器
    solver = diffrax.Dopri5()
    if adaptive:
        stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-9)
    else:
        stepsize_controller = diffrax.ConstantStepSize()
    ts = u.math.linspace(t0, t1, 1000)

    # 求解 ODE
    solution = diffrax.diffeqsolve(
        terms=diffrax.ODETerm(navier_stokes_ode),
        t0=t0,
        t1=t1,
        dt0=1e-3 * unit_of_t,
        y0=state0,
        args=args,
        solver=solver,
        stepsize_controller=stepsize_controller,
        saveat=diffrax.SaveAt(ts=ts),
        max_steps=100000
    )

    # 提取结果
    omega_sol = solution.ys[0]
    psi_sol = solution.ys[1]

    # 更新流函数
    def update_psi(omega_sol, dx, dy):
        return jax.vmap(lambda t: solve_poisson(t, dx, dy))(omega_sol)

    psi_updated = update_psi(omega_sol, dx, dy)

    return omega_sol, psi_updated


@partial(jax.jit, static_argnums=(0, 1))
def _integrate_without_unit(n=64, adaptive: bool = True):
    Nx, Ny = n, n

    # 定义离散化操作
    def laplacian(phi, dx, dy):
        return (
            (jnp.roll(phi, -1, axis=0) - 2 * phi + jnp.roll(phi, 1, axis=0)) / dx ** 2 +
            (jnp.roll(phi, -1, axis=1) - 2 * phi + jnp.roll(phi, 1, axis=1)) / dy ** 2
        )

    def solve_poisson(omega, dx, dy):
        Nx, Ny = omega.shape
        kx = jnp.fft.fftfreq(Nx, d=dx) * 2 * jnp.pi
        ky = jnp.fft.fftfreq(Ny, d=dy) * 2 * jnp.pi
        Kx, Ky = jnp.meshgrid(kx, ky, indexing='ij')
        K_squared = Kx ** 2 + Ky ** 2
        K_squared = jnp.where(K_squared == 0, 1.0, K_squared)  # 避免除以零
        omega_hat = jnp.fft.fft2(omega)
        psi_hat = -omega_hat / K_squared
        psi = jnp.real(jnp.fft.ifft2(psi_hat))
        return psi

    def navier_stokes_ode(t, state, args):
        omega, psi = state
        dx, dy, nu, f = args

        # 计算速度场
        u = (jnp.roll(psi, -1, axis=1) - jnp.roll(psi, 1, axis=1)) / (2 * dy)
        v = -(jnp.roll(psi, -1, axis=0) - jnp.roll(psi, 1, axis=0)) / (2 * dx)

        # 计算涡度的对流项
        domega_dx = (jnp.roll(omega, -1, axis=0) - jnp.roll(omega, 1, axis=0)) / (2 * dx)
        domega_dy = (jnp.roll(omega, -1, axis=1) - jnp.roll(omega, 1, axis=1)) / (2 * dy)
        advect = u * domega_dx + v * domega_dy

        # 计算涡度的扩散项
        diffusion = nu * laplacian(omega, dx, dy)

        # 计算外力的涡度贡献
        f_vorticity = ((jnp.roll(f[1], -1, axis=0) - jnp.roll(f[1], 1, axis=0)) / (2 * dx) -
                       (jnp.roll(f[0], -1, axis=1) - jnp.roll(f[0], 1, axis=1)) / (2 * dy))

        # d(omega)/dt
        domega_dt = -advect + diffusion + f_vorticity

        # 由于 psi 是通过泊松方程即时更新的，dpsi_dt 为零
        dpsi_dt = jnp.zeros_like(psi)

        return (domega_dt, dpsi_dt)

    def initial_conditions(Nx, Ny, dx, dy):
        # 示例：初始化为一个中心涡旋
        x = jnp.linspace(0, 2 * jnp.pi, Nx, endpoint=False)
        y = jnp.linspace(0, 2 * jnp.pi, Ny, endpoint=False)
        X, Y = jnp.meshgrid(x, y, indexing='ij')
        omega0 = jnp.sin(X) * jnp.sin(Y)
        psi0 = solve_poisson(omega0, dx, dy)
        return omega0, psi0

    # 设置网格和物理参数
    Lx, Ly = 2.0 * jnp.pi, 2.0 * jnp.pi
    dx, dy = Lx / Nx, Ly / Ny
    nu = 1e-3
    f = (jnp.zeros((Nx, Ny)), jnp.zeros((Nx, Ny)))  # 无外力

    # 初始条件
    state0 = initial_conditions(Nx, Ny, dx, dy)

    # 时间范围
    t0, t1 = 0.0, 10.0

    # ODE 参数
    args = (dx, dy, nu, f)

    # 定义 ODE 系统
    ode_system = lambda t, state, args: navier_stokes_ode(t, state, args)

    # 创建求解器
    solver = diffrax.Dopri5()
    if adaptive:
        stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-9)
    else:
        stepsize_controller = diffrax.ConstantStepSize()
    ts = jnp.linspace(t0, t1, 1000)

    # 自定义求解器步骤，以在每个时间步后更新 psi
    def coupled_ode_system(t, state, args):
        omega, psi = state
        domega_dt, dpsi_dt = ode_system(t, state, args)
        return (domega_dt, dpsi_dt)

    # 求解 ODE
    solution = diffrax.diffeqsolve(
        terms=diffrax.ODETerm(coupled_ode_system),
        t0=t0,
        t1=t1,
        dt0=1e-3,
        y0=state0,
        args=args,
        solver=solver,
        stepsize_controller=stepsize_controller,
        saveat=diffrax.SaveAt(ts=ts),
        max_steps=100000
    )

    # 提取结果
    omega_sol = solution.ys[0]
    psi_sol = solution.ys[1]

    # 更新流函数
    def update_psi(omega_sol, dx, dy):
        return jax.vmap(lambda t: solve_poisson(t, dx, dy))(omega_sol)

    psi_updated = update_psi(omega_sol, dx, dy)

    return omega_sol, psi_updated


def show_with_unit():
    omega_sol, psi_updated = _integrate_with_unit()

    # 选择时间步
    time_index = -1  # 最后一个时间步

    # 提取涡度和流函数
    omega_final = omega_sol[time_index]
    psi_final = psi_updated[time_index]

    # 绘制涡度场
    plt.figure(figsize=(6, 5))
    plt.contourf(omega_final.mantissa, levels=50, cmap='viridis')
    plt.colorbar(label='Vorticity')
    plt.title(f'Final Vorticity Field [{omega_final.unit}]\n(simulation with units)')
    plt.xlabel(f'X [{unit_of_x}]')
    plt.ylabel(f'Y [{unit_of_y}]')
    # plt.show()


def show_without_unit():
    omega_sol, psi_updated = _integrate_without_unit()

    # 选择时间步
    time_index = -1  # 最后一个时间步

    # 提取涡度和流函数
    omega_final = omega_sol[time_index]
    psi_final = psi_updated[time_index]

    # 绘制涡度场
    plt.figure(figsize=(6, 5))
    plt.contourf(omega_final, levels=50, cmap='viridis')
    plt.colorbar(label='Vorticity')
    plt.title('Final Vorticity Field\n(simulation without units)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


def compare_simulation_results():
    show_with_unit()
    show_without_unit()


scales = [40, 100, 200, 400]


def compare_compilation_time():
    import statistics
    from scipy import stats
    import pandas as pd

    platform = jax.default_backend()

    n_run = 10

    heads = ['n', 'unit_or_not', 'time']
    results = []

    for n in scales:
        with_unit_compile_times = []
        for _ in range(n_run):
            t0 = time.time()
            _integrate_with_unit.lower(n).compile()
            t1 = time.time()
            with_unit_compile_times.append(t1 - t0)

            jax.clear_caches()
            eqx.clear_caches()

            results.append([n * n, 'with unit', t1 - t0])

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

            results.append([n * n, 'without unit', t1 - t0])

        without_unit_mean = statistics.mean(without_unit_compile_times)
        without_unit_std = statistics.stdev(without_unit_compile_times)
        print(f'with unit compile time = {without_unit_mean} ± {without_unit_std} s')

        tstat, pval = stats.ttest_ind(a=with_unit_compile_times, b=without_unit_compile_times, alternative="two-sided")
        print("t-stat: {:.2f}   pval: {:.4f}".format(tstat, pval))
        print()

    pd.DataFrame(results, columns=heads).to_csv(f'results/Navier-Stokes-2d-{platform}-compile-time.csv', index=False)


def compare_simulation_time():
    import statistics
    from scipy import stats
    import pandas as pd

    platform = jax.default_backend()

    heads = ['n', 'unit_or_not', 'time']
    results = []
    n_run = [100, 100, 10, 10, 5]
    n_run = [100, 100, 100, 100, 100]
    n_run = [10, 10, 10, 10, 10]

    for i, n in enumerate(scales):

        jax.block_until_ready(_integrate_with_unit(n, False))  # compile the model with units
        with_unit_simulate_times = []
        for _ in range(n_run[i]):
            t0 = time.time()
            jax.block_until_ready(_integrate_with_unit(n, False))
            t1 = time.time()
            with_unit_simulate_times.append(t1 - t0)

            results.append([n * n, 'with unit', t1 - t0])

        print(f'with unit simulation time = {statistics.mean(with_unit_simulate_times)} '
              f'± {statistics.stdev(with_unit_simulate_times)} s')

        without_unit_simulate_times = []
        jax.block_until_ready(_integrate_without_unit(n, False))  # compile the model without units
        for _ in range(n_run[i]):
            t0 = time.time()
            jax.block_until_ready(_integrate_without_unit(n, False))
            t1 = time.time()
            without_unit_simulate_times.append(t1 - t0)

            results.append([n * n, 'without unit', t1 - t0])

        print(f'without unit simulation time = {statistics.mean(without_unit_simulate_times)} '
              f'± {statistics.stdev(without_unit_simulate_times)} s')

        tstat, pval = stats.ttest_ind(a=with_unit_simulate_times,
                                      b=without_unit_simulate_times,
                                      alternative="two-sided")
        print("t-stat: {:.2f}   pval: {:.4f}".format(tstat, pval))
        print()

        jax.clear_caches()
        eqx.clear_caches()

    pd.DataFrame(results, columns=heads).to_csv(f'results/Navier-Stokes-2d-{platform}-simulation-time.csv', index=False)


if __name__ == '__main__':
    # compare_simulation_results()
    # compare_compilation_time()
    compare_simulation_time()
