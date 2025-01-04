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


def vector_field(t, y, args):
    prey, predator = y
    α, β, γ, δ = args
    d_prey = α * prey - β * prey * predator
    d_predator = -γ * predator + δ * prey * predator
    d_y = d_prey / u.month, d_predator / u.month
    return d_y


term = ODETerm(vector_field)
solver = Tsit5()
t0 = 0 * u.month
t1 = 40 * u.month
dt0 = 0.1 * u.month
y0 = (10.0, 10.0)
args = (0.1, 0.02, 0.4, 0.02)
saveat = SaveAt(ts=u.math.linspace(t0, t1, 1000), t1=True, t0=True)
sol = diffeqsolve(term, solver, t0, t1, dt0, y0, args=args, saveat=saveat, max_steps=100000)
