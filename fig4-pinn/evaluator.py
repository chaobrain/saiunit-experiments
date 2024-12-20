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
import pickle
import sys
import time

import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import jax

jax.config.update('jax_cpu_enable_async_dispatch', False)
# jax.config.update('jax_platform_name', 'cpu')

from typing import Sequence, Callable
import brainunit as u
from contextlib import contextmanager
import brainstate as bst
import pinnx


@contextmanager
def change_stdout():
    stdout = sys.stdout
    stderr = sys.stderr
    try:
        with open(os.devnull, 'w') as devnull:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
    finally:
        sys.stdout = stdout
        sys.stderr = stderr


class Trainer(pinnx.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._train_times = []

    def _train(self, iterations, display_every, batch_size, callbacks):
        for i in range(iterations):
            callbacks.on_epoch_begin()
            callbacks.on_batch_begin()

            # get data
            self.train_state.set_data_train(*self.problem.train_next_batch(batch_size))

            # train one batch
            t0 = time.time()
            self.fn_train_step(self.train_state.X_train, self.train_state.y_train, **self.train_state.Aux_train)
            t1 = time.time()

            self._train_times.append(t1 - t0)

            self.train_state.epoch += 1
            self.train_state.step += 1
            if self.train_state.step % display_every == 0 or i + 1 == iterations:
                self._test()

            callbacks.on_batch_end()
            callbacks.on_epoch_end()

            if self.stop_training:
                break


def eval(
    problem,
    n_point,
    unit: bool = True,
    n_train: int = 15000,
    external_trainable_variables: Sequence = None,
    **kwargs
):
    trainer = Trainer(
        problem,
        external_trainable_variables=external_trainable_variables
    )
    _, compile_time = trainer.compile(bst.optim.Adam(1e-3), measture_train_step_compile_time=True)
    trainer.train(iterations=n_train)


    loss_train = jax.tree.map(lambda *xs: u.math.asarray(xs), *trainer.loss_history.loss_train, is_leaf=u.math.is_quantity)
    loss_test = jax.tree.map(lambda *xs: u.math.asarray(xs), *trainer.loss_history.loss_test, is_leaf=u.math.is_quantity)

    return dict(
        n_point=n_point,
        with_unit_or_not='with unit' if unit else 'without unit',
        compile_time=compile_time,
        train_times=np.asarray(trainer._train_times),
        loss_train=loss_train,
        loss_test=loss_test,
        best_loss_train=trainer.train_state.best_loss_train,
        best_loss_test=trainer.train_state.best_loss_test,
    )


def scaling_experiments(
    name: str,
    solve_with_unit: Callable[[float], dict],
    solve_without_unit: Callable[[float], dict],
    scales: Sequence[float] = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
):
    platform = jax.default_backend()
    os.makedirs('results/', exist_ok=True)

    for scale in scales:
        print(f"scale: {scale}")
        # with change_stdout():
        result1 = eval(**solve_with_unit(scale))
        with open(f'results/{name}_{platform}_scaling={scale}_with_unit.pkl', 'wb') as f:
            pickle.dump(result1, f)
        print(f'with unit: {result1["best_loss_test"]}, {result1["compile_time"]}, {result1["train_times"].mean()}')
        result2 = eval(**solve_without_unit(scale))
        with open(f'results/{name}_{platform}_scaling={scale}_without_unit.pkl', 'wb') as f:
            pickle.dump(result2, f)
        print(f'without unit: {result2["best_loss_test"]}, {result2["compile_time"]}, {result2["train_times"].mean()}')
        print()
