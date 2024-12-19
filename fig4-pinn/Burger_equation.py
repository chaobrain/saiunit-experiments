import jax

jax.config.update('jax_platform_name', 'cpu')

import brainunit as u

import pinnx

import evaluator


def solve_problem_with_unit(scale=1.0):
    geometry = pinnx.geometry.GeometryXTime(
        geometry=pinnx.geometry.Interval(-1, 1.),
        timedomain=pinnx.geometry.TimeDomain(0, 0.99)
    ).to_dict_point(x=u.meter, t=u.second)

    uy = u.meter / u.second
    bc = pinnx.icbc.DirichletBC(lambda x: {'y': 0. * uy})
    ic = pinnx.icbc.IC(lambda x: {'y': -u.math.sin(u.math.pi * x['x'] / u.meter) * uy})

    v = 0.01 / u.math.pi * u.meter ** 2 / u.second

    def pde(x, y):
        jacobian = approximator.jacobian(x)
        hessian = approximator.hessian(x)
        dy_x = jacobian['y']['x']
        dy_t = jacobian['y']['t']
        dy_xx = hessian['y']['x']['x']
        residual = dy_t + y['y'] * dy_x - v * dy_xx
        return residual

    approximator = pinnx.nn.Model(
        pinnx.nn.DictToArray(x=u.meter, t=u.second),
        pinnx.nn.FNN(
            [geometry.dim] + [20] * 3 + [1],
            "tanh",
        ),
        pinnx.nn.ArrayToDict(y=uy)
    )

    num_domain = int(2540 * scale)
    num_boundary = int(80 * scale)
    num_initial = int(160 * scale)

    problem = pinnx.problem.TimePDE(
        geometry,
        pde,
        [bc, ic],
        approximator,
        num_domain=num_domain,
        num_boundary=num_boundary,
        num_initial=num_initial,
    )

    return {
        'problem': problem,
        'n_point': num_domain + num_boundary + num_initial,
        'unit': True,
        'n_train': 15000
    }


def solve_problem_without_unit(scale=1.0):
    geometry = pinnx.geometry.GeometryXTime(
        geometry=pinnx.geometry.Interval(-1, 1.),
        timedomain=pinnx.geometry.TimeDomain(0, 0.99)
    ).to_dict_point(x=None, t=None)

    bc = pinnx.icbc.DirichletBC(lambda x: {'y': 0.})
    ic = pinnx.icbc.IC(lambda x: {'y': -u.math.sin(u.math.pi * x['x'])})

    v = 0.01 / u.math.pi

    def pde(x, y):
        jacobian = approximator.jacobian(x)
        hessian = approximator.hessian(x)
        dy_x = jacobian['y']['x']
        dy_t = jacobian['y']['t']
        dy_xx = hessian['y']['x']['x']
        residual = dy_t + y['y'] * dy_x - v * dy_xx
        return residual

    approximator = pinnx.nn.Model(
        pinnx.nn.DictToArray(x=None, t=None),
        pinnx.nn.FNN(
            [geometry.dim] + [20] * 3 + [1],
            "tanh",
        ),
        pinnx.nn.ArrayToDict(y=None)
    )

    num_domain = int(2540 * scale)
    num_boundary = int(80 * scale)
    num_initial = int(160 * scale)

    problem = pinnx.problem.TimePDE(
        geometry,
        pde,
        [bc, ic],
        approximator,
        num_domain=num_domain,
        num_boundary=num_boundary,
        num_initial=num_initial,
    )

    return {
        'problem': problem,
        'n_point': num_domain + num_boundary + num_initial,
        'unit': False,
        'n_train': 15000
    }


if __name__ == '__main__':
    evaluator.scaling_experiments(
        'Burger_equation',
        solve_with_unit=solve_problem_with_unit,
        solve_without_unit=solve_problem_without_unit,
        scales=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
        num_exp=10
    )
