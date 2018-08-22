from fenics import *
from fenics_adjoint import *


class InitialConditions(Expression):
    def eval_cell(self, values, x, ufl_cell):
        if self.mf[ufl_cell.index]==1:
            values[0] = 1
        else:
            values[0] = 0


def initial_condition(V, mesh_config):
    subdomains = mesh_config["subdomains"]
    init = InitialConditions(degree=1)
    init.mf = subdomains
    return interpolate(init, V)


def bc(g):
    for g_i in g:
        g_i.vector()[:] = 1
    return g

if __name__ == "__main__":
    from forward_problem import initialize_mesh, load_control_values, save_control_values, generate_observations, functional, gradient
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', default=0.0001, type=float)
    parser.add_argument('--num', default=0, type=int)
    parser.add_argument('--beta', default=0.001, type=float)
    parser.add_argument('--noise', default=0.0, type=float)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--D', default=[1000, 2, 1], type=float, nargs=3)
    parser.add_argument('--mesh', default="mesh_invers_contrast.h5", type=str)
    parser.add_argument("--tau", nargs="+", type=float)
    parser.add_argument("--k", type=int)
    parser.add_argument("--obs-file", default="U.xdmf", type=str)
    parser.add_argument("--load-control-values-file", default=None, type=str)
    parser.add_argument("--save-control-values-file", default=None, type=str)
    parser.add_argument("--generate-observations", default=False, type=bool)
    Z = parser.parse_args()

    mesh_config = initialize_mesh(Z.mesh)

    V = FunctionSpace(mesh_config["mesh"], "CG", 1)

    # Diffusion coefficients
    D = {1: Constant(Z.D[0]), 2: Constant(Z.D[1]), 3: Constant(Z.D[2])}

    # Observation timepoints
    tau = Z.tau

    # Boundary conditions
    k = Z.k
    g = [Function(V) for _ in range(k)]

    if Z.generate_observations:
        ic = initial_condition(V, mesh_config)
        g = bc(g)
        generate_observations(mesh_config, V, D, g, ic, tau, Z.obs_file)
        exit()
    else:
        J = functional(mesh_config, V, D, g, tau, Z.obs_file, Z.alpha, Z.beta, None)
    ctrls = ([Control(D[i]) for i in range(1, 4)]
             + [Control(g_i) for g_i in g])

    Jhat = ReducedFunctional(J, ctrls)
    Jhat.optimize()

    if Z.load_control_values_file is not None:
        m = load_control_values(k, Z.load_control_values_file)
        Jhat(m)

    lb = [100.0, 0.2, 0.1]
    ub = [10000.0, 20.0, 10.0]

    for i in range(3, len(ctrls)):
        lb.append(0.0)
        ub.append(10.0)

    opt_ctrls = minimize(Jhat, method="L-BFGS-B", bounds=(lb, ub), options={"disp": True, "maxiter": 1000, "gtol": 1e-02})

    print("[Constant({}), Constant({}), Constant({})]".format(float(opt_ctrls[0]), float(opt_ctrls[1]), float(opt_ctrls[2])))
    if Z.save_control_values_file is not None:
        save_control_values(opt_ctrls, Z.save_control_values_file)


