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


def bc(g, V):
    l = len(g)
    for i in range(l):
        #p = (i + 1)/float(l)
        #if i > l/2:
        #    p = (l - i + 1)/l
        #expr = Expression("sin(x[0])*sin(x[0])*cos(x[1])*cos(x[1])*sin(pi*i)", i=i/float(l), degree=1)
        #k = interpolate(expr, V)
        g_i = g[i]
        #g_i.vector()[:] = k.vector()
        if i < l/2:
            g_i.vector()[:] = i
        else:
            g_i.vector()[:] = l - i
    return g


def bc_guess(g, obs_file, tau, k):
    d = Function(g[0].function_space())
    dt = tau[-1]/k
    obs_file = HDF5File(mpi_comm_world(), obs_file, 'r')
    next_tau = 0
    t = 0.0
    for i in range(k):
        t += dt
        obs_file.read(d, str(tau[next_tau]))
        g[i].vector()[:] = d.vector()[:]

        if abs(t - tau[next_tau]) < abs(t + dt - tau[next_tau]):
            next_tau += 1
    return g


iter_cnt = 0
def iter_cb(m):
    global iter_cnt
    iter_cnt += 1
    print("Coeffs-Iter: {} | Constant({}) | Constant({}) | Constant({})".format(iter_cnt, float(m[0]), float(m[1]), float(m[2])))
    from pyadjoint.reduced_functional_numpy import ReducedFunctionalNumPy
    NumRF = ReducedFunctionalNumPy(Jhat)
    ds = mesh_config["ds"]
    fenics_m = NumRF.set_local([control.copy_data() for control in Jhat.controls], m)
    print("DirichletBC-Iter: {} | {}".format(iter_cnt, sum([assemble((fenics_m[i] - correct_g[i-3])**2*ds(1)) for i in range(3, len(fenics_m))])/sum([assemble(correct_g[i-3]**2*ds(1)) for i in range(3, len(fenics_m)) ]) ))


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
    parser.add_argument("--maxiter", default=50, type=int)
    Z = parser.parse_args()
    print Z
    mesh_config = initialize_mesh(Z.mesh)

    V = FunctionSpace(mesh_config["mesh"], "CG", 1)

    # Diffusion coefficients
    D = {1: Constant(Z.D[0]), 2: Constant(Z.D[1]), 3: Constant(Z.D[2])}

    # Observation timepoints
    tau = Z.tau

    # Boundary conditions
    k = Z.k
    g = [Function(V) for _ in range(k)]
    correct_g = [Function(V) for _ in range(k)]
    correct_g = bc(correct_g, V)

    if Z.generate_observations:
        ic = initial_condition(V, mesh_config)
        g = bc(g, V)
        generate_observations(mesh_config, V, D, g, ic, tau, Z.obs_file)
        exit()
    else:
        g = bc_guess(g, Z.obs_file, tau, k)
        #g = bc(g)
        #for g_i in g:
        #    g_i.vector()[:] = 10
        J = functional(mesh_config, V, D, g, tau, Z.obs_file, Z.alpha, Z.beta, None)
        #exit()
    ctrls = ([Control(D[i]) for i in range(1, 4)]
             + [Control(g_i) for g_i in g])

    print("J = ", J)
    Jhat = ReducedFunctional(J, ctrls)
    Jhat.optimize()

    if Z.load_control_values_file is not None:
        m = load_control_values(k, V, Z.load_control_values_file)
        Jhat(m)

    lb = [100.0, 0.2, 0.1]
    ub = [10000.0, 20.0, 10.0]

    for i in range(len(lb), len(ctrls)):
        lb.append(-10.0)
        ub.append(100.0)

    opt_ctrls = minimize(Jhat, method="L-BFGS-B", bounds=(lb, ub), callback = iter_cb, options={"disp": True, "maxiter": Z.maxiter, "gtol": 1e-2})

    #print("[Constant({}), Constant({}), Constant({})]".format(float(opt_ctrls[0]), float(opt_ctrls[1]), float(opt_ctrls[2])))
    #print(
    #"[Constant({}), Constant({}))]".format(float(opt_ctrls[0]), float(opt_ctrls[1])))
    if Z.save_control_values_file is not None:
        save_control_values(opt_ctrls, Z.save_control_values_file)

