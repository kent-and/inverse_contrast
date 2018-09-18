from fenics import *
from fenics_adjoint import *


class InitialConditions(Expression):
    def eval_cell(self, values, x, ufl_cell):
        if self.mf[ufl_cell.index]==1:
            values[0] = 1
        else:
            values[0] = 0


def add_noise(noise):
     for i in noise:
         i.vector()[:]+=Z.noise*(0.5-random.random(i.vector().size()) )


def initial_condition(V, mesh_config):
    subdomains = mesh_config["subdomains"]
    init = InitialConditions(degree=1)
    init.mf = subdomains
    return interpolate(init, V)


def bc(g, V,tau,k, mesh_config):
    t = tau[0]
    dt = (tau[-1]-tau[0])/k
    Exp = Expression("A+B*t-C*t*t", A=0.3, B=0.167,C=0.007 , t=0.0 , degree=1)
    for i in range(k):
        Exp.t= t
        bc = DirichletBC(V, Exp, mesh_config["boundaries"], 1)
        bc.apply(g[i].vector())
        t+=dt
        
    return g


def bc_guess(g, obs_file, tau, k,noise):
    d = Function(g[0].function_space())
    dt = (tau[-1] -tau[0])/k
    obs_file = HDF5File(mpi_comm_world(), obs_file, 'r')
    next_tau = 1
    t = tau[0]
    for i in range(k):
        t += dt
        obs_file.read(d,"%0.2f"%(tau[next_tau]))
        g[i].vector()[:] = d.vector()[:]
        if noise:
           g[i].vector()[:]+= noise[next_tau].vector()[:]

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
    print("Functional-value: {} | {} ".format(iter_cnt, NumRF(m)) )
    print("DirichletBC-Iter: {} | {}".format(iter_cnt, sum([assemble((fenics_m[i] - correct_g[i-3])**2*ds) for i in range(3, len(fenics_m))])/sum([assemble(correct_g[i-3]**2*ds) for i in range(3, len(fenics_m)) ]) ))


if __name__ == "__main__":
    from forward_problem import initialize_mesh, load_control_values, save_control_values, generate_observations, functional
    from numpy import random
    import argparse
    import sys 
    print( sys.version )
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', default=0.0001, type=float)
    parser.add_argument('--num', default=0, type=int)
    parser.add_argument('--beta', default=0.001, type=float)
    parser.add_argument('--noise', default=0.0, type=float)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--D', default=[1000, 1, 1], type=float, nargs=3)
    parser.add_argument('--mesh', default="mesh_invers_contrast.h5", type=str)
    parser.add_argument("--tau", nargs="+", type=float)
    parser.add_argument("--k", type=int)
    parser.add_argument("--obs-file", default="U.xdmf", type=str)
    parser.add_argument("--load-control-values-file", default=None, type=str)
    parser.add_argument("--save-control-values-file", default=None, type=str)
    parser.add_argument("--generate-observations", default=False, type=bool)
    parser.add_argument("--maxiter", default=1000, type=int)
    parser.add_argument("--dx", default=0.0, type=float)
    Z = parser.parse_args()
    print(Z)
    mesh_config = initialize_mesh(Z.mesh)

    V = FunctionSpace(mesh_config["mesh"], "CG", 1)

    # Diffusion coefficients
    D = {1: Constant(Z.D[0]), 2: Constant(Z.D[1]), 3: Constant(Z.D[2])}

    # Observation timepoints
    if Z.dx!=0.0 : # dx can be simpler 
       k = int( (Z.tau[-1] - Z.tau[0])/dx)
       tau =[ Z.tau[0] + dx*i for i in range(k)]   
    else : 
       k = Z.k
       tau = Z.tau

    # Boundary conditions

    g = [Function(V) for _ in range(k)]
    correct_g = [Function(V) for _ in range(k)]
    correct_g = bc(correct_g, V,tau,k, mesh_config)
    # Noise 

    if Z.noise!=0.0:
       noise = [ Function(V) for _ in tau]
       add_noise(noise)
    else : 
       noise = None


    if Z.generate_observations:
        ic = initial_condition(V, mesh_config)
        g = bc(g, V,tau,k, mesh_config)
        generate_observations(mesh_config, V, D, g, ic, tau, Z.obs_file)
        exit()
    else:
        g = bc_guess(g, Z.obs_file, tau, k, noise)
        J = functional(mesh_config, V, D, g, tau, Z.obs_file, Z.alpha, Z.beta,None, noise=noise)

    ctrls = ([Control(D[i]) for i in range(1, 4)]
             + [Control(g_i) for g_i in g])

    print("J = ", J)
    Jhat = ReducedFunctional(J, ctrls)
    
    Jhat.optimize()

    
    opt_ctrls = minimize(Jhat, method="L-BFGS-B", callback = iter_cb, options={"disp": True, "maxiter": Z.maxiter, "gtol": 1.0e-1})

    #print("[Constant({}), Constant({}), Constant({})]".format(float(opt_ctrls[0]), float(opt_ctrls[1]), float(opt_ctrls[2])))
    #print(
    #"[Constant({}), Constant({}))]".format(float(opt_ctrls[0]), float(opt_ctrls[1])))
    if Z.save_control_values_file is not None:
        save_control_values(opt_ctrls, Z.save_control_values_file)


