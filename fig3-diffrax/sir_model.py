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


import brainunit as u

from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5


β, γ = 0.2 / u.day, 0.1 / u.day

def vf(t, y, args):
    S, I, R = y
    dS = -β * S * I
    dI = β * S * I - γ * I
    dR = γ * I
    return dS, dI, dR


term = ODETerm(vf)
solver = Tsit5()
t0 = 0 * u.day
t1 = 20 * u.day
dt0 = 0.1 * u.day
y0 = (0.99, 0.01, 0.0)
saveat = SaveAt(ts=u.math.linspace(t0, t1, 1000), t1=True, t0=True)
sol = diffeqsolve(term, solver, t0, t1, dt0, y0, saveat=saveat)
