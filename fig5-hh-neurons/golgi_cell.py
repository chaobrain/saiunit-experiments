# Copyright 2025 BDP Ecosystem Limited. All Rights Reserved.
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

# -*- coding: utf-8 -*-


import brainstate
import saiunit as u
import braincell


class ICav31_Ma2020(braincell.channel.CalciumChannel):
    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max=2.5e-4 * (u.cm / u.second),
        V_sh=0 * u.mV,
        T_base: brainstate.typing.ArrayLike = 3,
        T: brainstate.typing.ArrayLike = 22.,
    ):
        super().__init__(size=size)

        # parameters
        self.g_max = brainstate.init.param(g_max, self.varshape)
        self.T = brainstate.init.param(T, self.varshape)
        self.T_base = brainstate.init.param(T_base, self.varshape)
        self.V_sh = brainstate.init.param(V_sh, self.varshape)
        self.phi = brainstate.init.param(T_base((T - 37) / 10), self.varshape)

        self.v0_m_inf = -52 * u.mV
        self.v0_h_inf = -72 * u.mV
        self.k_m_inf = -5 * u.mV
        self.k_h_inf = 7 * u.mV

        self.C_tau_m = 1
        self.A_tau_m = 1.0
        self.v0_tau_m1 = -40 * u.mV
        self.v0_tau_m2 = -102 * u.mV
        self.k_tau_m1 = 9 * u.mV
        self.k_tau_m2 = -18 * u.mV

        self.C_tau_h = 15
        self.A_tau_h = 1.0
        self.v0_tau_h1 = -32 * u.mV
        self.k_tau_h1 = 7 * u.mV
        self.z = 2

    def init_state(self, V, Ca: braincell.IonInfo):
        self.p = braincell.DiffEqState(brainstate.init.param(u.math.zeros, self.varshape))
        self.q = braincell.DiffEqState(brainstate.init.param(u.math.zeros, self.varshape))

    def reset_state(self, V, Ca):
        self.p.value = self.f_p_inf(V)
        self.q.value = self.f_q_inf(V)

    def compute_derivative(self, V, Ca):
        self.p.derivative = self.phi * (self.f_p_inf(V) - self.p.value) / self.f_p_tau(V) / u.ms
        self.q.derivative = self.phi * (self.f_q_inf(V) - self.q.value) / self.f_q_tau(V) / u.ms

    def f_p_inf(self, V):
        return 1 / (1 + u.math.exp((V - self.v0_m_inf) / self.k_m_inf))

    def f_q_inf(self, V):
        return 1 / (1 + u.math.exp((V - self.v0_h_inf) / self.k_h_inf))

    def f_p_tau(self, V):
        return u.math.where(
            V <= -90 * u.mV,
            1.,
            (self.C_tau_m +
             self.A_tau_m / (
                 u.math.exp((V - self.v0_tau_m1) / self.k_tau_m1) +
                 u.math.exp((V - self.v0_tau_m2) / self.k_tau_m2)
             ))
        )

    def f_q_tau(self, V):
        return self.A_tau_h / u.math.exp((V - self.v0_tau_h1) / self.k_tau_h1) + self.C_tau_h

    def ghk(self, V, ci, co=2 * u.mM):
        F = u.faraday_constant
        R = u.gas_constant
        zeta = (self.z * F * V) / (R * u.celsius2kelvin(self.T))
        g1 = (self.z * F) * (ci - co * u.math.exp(-zeta)) * (1 + zeta / 2)
        g2 = (self.z * zeta * F) * (ci - co * u.math.exp(-zeta)) / (1 - u.math.exp(-zeta))
        cond = u.math.abs((1 - u.math.exp(-zeta))) <= 1e-6
        return u.math.where(cond, g1, g2)

    def current(self, V, Ca: braincell.IonInfo):
        m = self.p.value
        h = self.q.value
        return -self.g_max * m ** 2 * h * self.ghk(V, Ca.C)
