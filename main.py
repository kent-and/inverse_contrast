from fenics import *
from fenics_adjoint import *


class InitialConditions(Expression):
    def eval_cell(self, values, x, ufl_cell):
        if self.mf[ufl_cell.index]==1:
            values[0] = 1
        else:
            values[0] = 0


def add_noise(noise,amp): 
     from numpy import random
     for i in noise:
         i.vector().add_local( amp*(0.5-random.random(i.vector().local_size()) ) )


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



def contains_observation(t,tau,dt,next_tau):
 
    if abs(t - tau[next_tau]) < abs(t + dt - tau[next_tau]):  
       next_tau = contains_observation(t,tau,dt,next_tau+1)

    return next_tau

def bc_guess(g, obs_file, tau, k,noise):
    #d = Function(g[0].function_space()) # Needs more presentable code
    
    dt = (tau[-1] -tau[0])/k
    obs_file = HDF5File(mpi_comm_world(), obs_file, 'r')
    next_tau = 1
    t = tau[0]
    for i in range(k):
        t += dt
        obs_file.read(g[i],"%0.2f"%(tau[next_tau]))
        #g[i].vector()[:] = d.vector()[:]

        if noise:
           g[i].vector()[:]+= noise[next_tau].vector()[:]

        next_tau = contains_observation(t,tau,dt,next_tau)


    return g


iter_cnt = 0
def iter_cb(m):
    global iter_cnt
    iter_cnt += 1
    print("Coeffs-Iter: {} | Constant({}) | Constant({}) ".format(iter_cnt, float(m[0]), float(m[1])))



if __name__ == "__main__":
    from forward_problem import initialize_mesh, load_control_values, save_control_values, generate_observations, functional
    import argparse
    import sys 
    print( sys.version )
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', default=0.0001, type=float)
    parser.add_argument('--num', default=0, type=int)
    parser.add_argument('--beta', default=0.001, type=float)
    parser.add_argument('--noise', default=0.0, type=float)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--D', default=[1, 1], type=float, nargs=2)
    parser.add_argument('--scale', default=[1, 1], type=float, nargs=2)
    parser.add_argument('--mesh', default="mesh2domain.h5", type=str)
    parser.add_argument("--tau", nargs="+", type=float)
    parser.add_argument("--k",default=1, type=int)
    parser.add_argument("--K",default=0, type=int)
    parser.add_argument("--obs-file", default="U.xdmf", type=str)
    parser.add_argument("--load-control-values-file", default=None, type=str)
    parser.add_argument("--save-control-values-file", default=None, type=str)
    parser.add_argument("--generate-observations", default=False, type=bool)
    parser.add_argument("--maxiter", default=1000, type=int)
    parser.add_argument("--dx", default=0.0, type=float) 
    parser.add_argument("--save",default=False, type=bool)
    parser.add_argument("--T1map", default=None,type=str)
    
    Z = parser.parse_args()
    print(Z)
    mesh_config = initialize_mesh(Z.mesh)
    
    V = FunctionSpace(mesh_config["mesh"], "CG", 1)

    # Diffusion coefficients
    D = {1: Constant(Z.D[0]), 2: Constant(Z.D[1])}

    tau = Z.tau
    # Number timepoints
    if Z.K==0:
      k = int(len(tau)-1)*Z.k
    else :
      k =Z.K
    # Boundary conditions
      
    g = [Function(V) for _ in range(k)]

    # Noise 

    if Z.noise!=0:
       noise = [ Function(V) for _ in tau]
       add_noise(noise,Z.noise)
    else : 
       noise = None
  
    if Z.save:
       save = File( "Results-{}-{}-{}-{}/observation.pvd".format( Z.alpha, Z.beta,Z.noise,k ) )
    else:
       save = None

    if Z.generate_observations:
        ic = initial_condition(V, mesh_config)
        g = bc(g, V,tau,k, mesh_config)
        generate_observations(mesh_config, V, D, g, ic, tau, Z.obs_file)
        exit()
    else:
        g = bc_guess(g, Z.obs_file, tau, k, noise)
        J = functional(mesh_config, V, D, g, tau, Z.obs_file, Z.alpha, Z.beta, noise=noise, gradient=Z.scale, save=save)

    ctrls = ([Control(D[i]) for i in range(1, 3)]
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


    if Z.save:
       tape = get_working_tape()

       from fenics_adjoint.solving import SolveBlock
       s_blocks = [block for block in tape._blocks if isinstance(block, SolveBlock)]
       states = [block.get_outputs()[0].saved_output for block in s_blocks]
       out =  File("Results-{}-{}-{}-{}/finalstate.pvd".format(Z.alpha,Z.beta, Z.noise,k) )

       for i in states:
           out << i

