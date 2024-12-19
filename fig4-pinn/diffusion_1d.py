import brainunit as u
import pinnx

import evaluator


def solve_problem_with_unit(scale: float = 1.0):
    unit_of_x = u.meter
    unit_of_t = u.second
    unit_of_f = 1 / u.second

    c = 1. * u.meter ** 2 / u.second

    geom = pinnx.geometry.Interval(-1, 1)
    timedomain = pinnx.geometry.TimeDomain(0, 1)
    geomtime = pinnx.geometry.GeometryXTime(geom, timedomain)
    geomtime = geomtime.to_dict_point(x=unit_of_x, t=unit_of_t)

    def func(x):
        y = u.math.sin(u.math.pi * x['x'] / unit_of_x) * u.math.exp(-x['t'] / unit_of_t)
        return {'y': y}

    bc = pinnx.icbc.DirichletBC(func)
    ic = pinnx.icbc.IC(func)

    net = pinnx.nn.Model(
        pinnx.nn.DictToArray(x=unit_of_x, t=unit_of_t),
        pinnx.nn.FNN([2] + [32] * 3 + [1], "tanh"),
        pinnx.nn.ArrayToDict(y=None),
    )

    def pde(x, y):
        jacobian = net.jacobian(x, x='t')
        hessian = net.hessian(x, xi='x', xj='x')
        dy_t = jacobian["y"]["t"]
        dy_xx = hessian["y"]["x"]["x"]
        source = (
            u.math.exp(-x['t'] / unit_of_t) * (
            u.math.sin(u.math.pi * x['x'] / unit_of_x) -
            u.math.pi ** 2 * u.math.sin(u.math.pi * x['x'] / unit_of_x)
        )
        )
        return dy_t - c * dy_xx + source * unit_of_f

    num_domain = int(40 * scale)
    num_boundary = int(20 * scale)
    num_initial = int(10 * scale)
    num_test = int(100 * scale)

    problem = pinnx.problem.TimePDE(
        geomtime,
        pde,
        [bc, ic],
        net,
        num_domain=num_domain,
        num_boundary=num_boundary,
        num_initial=num_initial,
        solution=func,
        num_test=num_test,
    )

    return {
        'problem': problem,
        'n_point': num_domain + num_boundary + num_initial,
        'unit': True,
        'n_train': 10000
    }


def solve_problem_without_unit(scale: float = 1.0):
    c = 1.

    geom = pinnx.geometry.Interval(-1, 1)
    timedomain = pinnx.geometry.TimeDomain(0, 1)
    geomtime = pinnx.geometry.GeometryXTime(geom, timedomain)
    geomtime = geomtime.to_dict_point(x=None, t=None)

    def func(x):
        y = u.math.sin(u.math.pi * x['x']) * u.math.exp(-x['t'])
        return {'y': y}

    bc = pinnx.icbc.DirichletBC(func)
    ic = pinnx.icbc.IC(func)

    net = pinnx.nn.Model(
        pinnx.nn.DictToArray(x=None, t=None),
        pinnx.nn.FNN([2] + [32] * 3 + [1], "tanh"),
        pinnx.nn.ArrayToDict(y=None),
    )

    def pde(x, y):
        jacobian = net.jacobian(x, x='t')
        hessian = net.hessian(x, xi='x', xj='x')
        dy_t = jacobian["y"]["t"]
        dy_xx = hessian["y"]["x"]["x"]
        source = (
            u.math.exp(-x['t']) * (
            u.math.sin(u.math.pi * x['x']) -
            u.math.pi ** 2 * u.math.sin(u.math.pi * x['x'])
        )
        )
        return dy_t - c * dy_xx + source

    num_domain = int(40 * scale)
    num_boundary = int(20 * scale)
    num_initial = int(10 * scale)
    num_test = int(100 * scale)

    problem = pinnx.problem.TimePDE(
        geomtime,
        pde,
        [bc, ic],
        net,
        num_domain=num_domain,
        num_boundary=num_boundary,
        num_initial=num_initial,
        solution=func,
        num_test=num_test,
    )

    return {
        'problem': problem,
        'n_point': num_domain + num_boundary + num_initial,
        'unit': False,
        'n_train': 10000
    }


if __name__ == '__main__':
    evaluator.scaling_experiments(
        'diffusion_1d',
        solve_with_unit=solve_problem_with_unit,
        solve_without_unit=solve_problem_without_unit,
        scales=(1.0, 2.0, 5.0, 10.0),
        num_exp=10
    )
