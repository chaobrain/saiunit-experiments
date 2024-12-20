import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import jax

jax.config.update('jax_cpu_enable_async_dispatch', False)
jax.config.update('jax_platform_name', 'cpu')

import brainstate as bst
import brainunit as u
import numpy as np
import pinnx

import evaluator


def solve_problem_with_unit(scale: float = 1.0):
    unit_of_x = u.meter
    unit_of_t = u.second
    unit_of_c = u.mole / u.meter ** 3

    kf = bst.ParamState(0.05 * u.meter ** 6 / u.mole ** 2 / u.second)
    D = bst.ParamState(1.0 * u.meter ** 2 / u.second)

    def pde(x, y):
        jacobian = net.jacobian(x, x='t')
        hessian = net.hessian(x)
        ca, cb = y['ca'], y['cb']
        dca_t = jacobian['ca']['t']
        dcb_t = jacobian['cb']['t']
        dca_xx = hessian['ca']['x']['x']
        dcb_xx = hessian['cb']['x']['x']
        eq_a = dca_t - 1e-3 * D.value * dca_xx + kf.value * ca * cb ** 2
        eq_b = dcb_t - 1e-3 * D.value * dcb_xx + 2 * kf.value * ca * cb ** 2
        return [eq_a, eq_b]

    net = pinnx.nn.Model(
        pinnx.nn.DictToArray(x=unit_of_x, t=unit_of_t),
        pinnx.nn.FNN([2] + [20] * 3 + [2], "tanh"),
        pinnx.nn.ArrayToDict(ca=unit_of_c, cb=unit_of_c),
    )

    geom = pinnx.geometry.Interval(0, 1)
    timedomain = pinnx.geometry.TimeDomain(0, 10)
    geomtime = pinnx.geometry.GeometryXTime(geom, timedomain)
    geomtime = geomtime.to_dict_point(x=unit_of_x, t=unit_of_t)

    def fun_bc(x):
        c = (1 - x['x'] / unit_of_x) * unit_of_c
        return {'ca': c, 'cb': c}

    bc = pinnx.icbc.DirichletBC(fun_bc)

    def fun_init(x):
        return {
            'ca': u.math.exp(-20 * x['x'] / unit_of_x) * unit_of_c,
            'cb': u.math.exp(-20 * x['x'] / unit_of_x) * unit_of_c,
        }

    ic = pinnx.icbc.IC(fun_init)

    def gen_traindata():
        data = np.load("./dataset/reaction.npz")
        t, x, ca, cb = data["t"], data["x"], data["Ca"], data["Cb"]
        X, T = np.meshgrid(x, t)
        x = {'x': X.flatten() * unit_of_x, 't': T.flatten() * unit_of_t}
        y = {'ca': ca.flatten() * unit_of_c, 'cb': cb.flatten() * unit_of_c}
        return x, y

    observe_x, observe_y = gen_traindata()
    observe_bc = pinnx.icbc.PointSetBC(observe_x, observe_y)

    num_domain = int(2000 * scale)
    num_boundary = int(100 * scale)
    num_initial = int(100 * scale)
    num_test = int(500 * scale)

    data = pinnx.problem.TimePDE(
        geomtime,
        pde,
        [bc, ic, observe_bc],
        net,
        num_domain=num_domain,
        num_boundary=num_boundary,
        num_initial=num_initial,
        num_test=num_test,
        anchors=observe_x,
    )

    return {
        'problem': data,
        'n_point': num_domain + num_boundary + num_initial,
        'unit': True,
        'n_train': 15000,
        'external_trainable_variables': [kf, D]
    }


def solve_problem_without_unit(scale: float = 1.0):
    kf = bst.ParamState(0.05)
    D = bst.ParamState(1.0)

    def pde(x, y):
        jacobian = net.jacobian(x, x='t')
        hessian = net.hessian(x)
        ca, cb = y['ca'], y['cb']
        dca_t = jacobian['ca']['t']
        dcb_t = jacobian['cb']['t']
        dca_xx = hessian['ca']['x']['x']
        dcb_xx = hessian['cb']['x']['x']
        eq_a = dca_t - 1e-3 * D.value * dca_xx + kf.value * ca * cb ** 2
        eq_b = dcb_t - 1e-3 * D.value * dcb_xx + 2 * kf.value * ca * cb ** 2
        return [eq_a, eq_b]

    net = pinnx.nn.Model(
        pinnx.nn.DictToArray(x=None, t=None),
        pinnx.nn.FNN([2] + [20] * 3 + [2], "tanh"),
        pinnx.nn.ArrayToDict(ca=None, cb=None),
    )

    geom = pinnx.geometry.Interval(0, 1)
    timedomain = pinnx.geometry.TimeDomain(0, 10)
    geomtime = pinnx.geometry.GeometryXTime(geom, timedomain)
    geomtime = geomtime.to_dict_point(x=None, t=None)

    def fun_bc(x):
        c = (1 - x['x'])
        return {'ca': c, 'cb': c}

    bc = pinnx.icbc.DirichletBC(fun_bc)

    def fun_init(x):
        return {
            'ca': u.math.exp(-20 * x['x']),
            'cb': u.math.exp(-20 * x['x']),
        }

    ic = pinnx.icbc.IC(fun_init)

    def gen_traindata():
        data = np.load("./dataset/reaction.npz")
        t, x, ca, cb = data["t"], data["x"], data["Ca"], data["Cb"]
        X, T = np.meshgrid(x, t)
        x = {'x': X.flatten(), 't': T.flatten()}
        y = {'ca': ca.flatten(), 'cb': cb.flatten()}
        return x, y

    observe_x, observe_y = gen_traindata()
    observe_bc = pinnx.icbc.PointSetBC(observe_x, observe_y)

    num_domain = int(2000 * scale)
    num_boundary = int(100 * scale)
    num_initial = int(100 * scale)
    num_test = int(500 * scale)

    data = pinnx.problem.TimePDE(
        geomtime,
        pde,
        [bc, ic, observe_bc],
        net,
        num_domain=num_domain,
        num_boundary=num_boundary,
        num_initial=num_initial,
        num_test=num_test,
        anchors=observe_x,
    )

    return {
        'problem': data,
        'n_point': num_domain + num_boundary + num_initial,
        'unit': True,
        'n_train': 15000,
        'external_trainable_variables': [kf, D]
    }


if __name__ == '__main__':
    evaluator.scaling_experiments(
        'diffusion_2d',
        solve_with_unit=solve_problem_with_unit,
        solve_without_unit=solve_problem_without_unit,
        scales=(1.0, 2.0, 5.0, 10.0),
        # num_exp=10
    )
