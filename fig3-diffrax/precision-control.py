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


import braintools
import brainunit as u
import matplotlib.pyplot as plt
from diffrax import Dopri5, Tsit5
from diffrax import diffeqsolve, ODETerm, SaveAt


def chemical_kinetic_first_order():
    k = 0.2 / u.ms
    y0 = 0.50 * u.molar

    t0 = 0 * u.ms
    t1 = 10 * u.ms

    saveat = SaveAt(ts=u.math.linspace(t0, t1, 100), t1=True, t0=True)
    sol = diffeqsolve(
        ODETerm(lambda t, y, args: -k * y),
        Dopri5(),
        t0=t0,
        t1=t1,
        dt0=0.01 * u.ms,
        y0=y0,
        saveat=saveat
    )

    # plt.plot(sol.ts, sol.ys, label="Concentration")
    # plt.legend()
    # plt.show()

    # sns.set_theme(font_scale=1.1)
    fig, gs = braintools.visualize.get_figure(1, 1, 3, 4)
    ax = fig.add_subplot(gs[0, 0])
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.plot(sol.ts, sol.ys, label="A")
    plt.ylabel("Concentration (Molar)")
    plt.xlim(t0, t1)
    plt.legend(fontsize=10, bbox_to_anchor=(1.0, 1.0), loc="upper left")
    plt.savefig("results/chemical_kinetic_first_order.svg")
    # plt.show()


def sir_model():
    N = 10000
    N = 1
    scale = u.day
    β, γ = 0.3 / scale, 0.1 / scale
    β, γ = 0.2 / scale, 0.1 / scale

    # β, γ = 0.1 / scale, 0.2 / scale
    # β, γ = 1.5 / scale, 0.1 / scale

    def vf(t, y, args):
        S, I, R = y
        dS = -β * S * I
        dI = β * S * I - γ * I
        dR = γ * I
        return dS, dI, dR

    term = ODETerm(vf)
    solver = Tsit5()
    t0 = 0 * scale
    t1 = 20 * scale
    dt0 = 0.1 * scale
    y0 = (0.99, 0.01, 0.0)
    saveat = SaveAt(ts=u.math.linspace(t0, t1, 1000), t1=True, t0=True)
    sol = diffeqsolve(term, solver, t0, t1, dt0, y0, saveat=saveat)

    # sns.set_theme(font_scale=1.1)
    fig, gs = braintools.visualize.get_figure(1, 1, 3, 4)
    ax = fig.add_subplot(gs[0, 0])
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.plot(sol.ts, sol.ys[0] * N, label="S")
    plt.plot(sol.ts, sol.ys[1] * N, label="I")
    plt.plot(sol.ts, sol.ys[2] * N, label="R")
    # plt.ticklabel_format(axis='y', style='sci', scilimits=(1, 2))
    # plt.title("SIR Model")
    plt.ylabel(r"Population Size ($\times  10^4$)")
    # plt.ylabel("Population Size")
    plt.xlim(t0, t1)
    plt.legend(fontsize=10, bbox_to_anchor=(1.0, 1.0), loc="upper left")
    plt.savefig("results/sir_model.svg")
    # plt.show()


def lotka_volterra_equation():
    scale = u.month

    def vector_field(t, y, args):
        prey, predator = y
        α, β, γ, δ = args
        d_prey = α * prey - β * prey * predator
        d_predator = -γ * predator + δ * prey * predator
        d_y = d_prey / scale, d_predator / scale
        return d_y

    term = ODETerm(vector_field)
    solver = Tsit5()
    t0 = 0 * scale
    t1 = 40 * scale
    dt0 = 0.1 * scale
    y0 = (10.0, 10.0)
    args = (0.1, 0.02, 0.4, 0.02)
    saveat = SaveAt(ts=u.math.linspace(t0, t1, 1000), t1=True, t0=True)
    sol = diffeqsolve(term, solver, t0, t1, dt0, y0, args=args, saveat=saveat, max_steps=100000)

    # sns.set_theme(font_scale=1.1)
    fig, gs = braintools.visualize.get_figure(1, 1, 3, 4)
    ax = fig.add_subplot(gs[0, 0])
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.plot(sol.ts, sol.ys[0], label="R")
    plt.plot(sol.ts, sol.ys[1], label="P")
    # plt.title("Lotka-Volterra Equation")
    plt.ylabel("Population Size")
    plt.xlim(t0, t1)
    # plt.legend(fontsize=10)
    plt.legend(fontsize=10, bbox_to_anchor=(1.0, 1.0), loc="upper left")
    plt.savefig("results/lotka_volterra_equation.svg")
    # plt.show()


if __name__ == '__main__':
    pass
    # chemical_kinetic_zero()
    chemical_kinetic_first_order()
    sir_model()
    lotka_volterra_equation()
