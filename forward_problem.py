# -*- coding: utf-8 -*-
from dolfin import *
from fenics_adjoint import *

set_log_level(INFO)


def initialize_mesh(mesh_file): # Lars : Endret 
    # Import mesh and subdomains
    mesh = Mesh()

    hdf = HDF5File(mesh.mpi_comm(), "mesh_invers_contrast.h5", "r")
    hdf.read(mesh, "/mesh", False, annotate=False)
    subdomains = MeshFunction("size_t", mesh, mesh.topology().dim())
    hdf.read(subdomains, "/subdomains", annotate=False)
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    hdf.read(boundaries, "/boundaries", annotate=False)
    # Define measures with subdomains
    dx = Measure("dx", domain=mesh, subdomain_data=subdomains)
    ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
    return {"ds": ds, "dx": dx, "boundaries": boundaries, "mesh": mesh, "subdomains": subdomains}


def forward_problem(context):
    V = context.V
    # Define trial and test-functions
    u = TrialFunction(V)
    v = TestFunction(V)

    # Solution at current and previous time
    U_prev = Function(V)
    U = context.ic

    dt = context.dt
    D = context.D
    dx = context.dx

    # Define bilinear form, handling each subdomain 1, 2, and 3 in separate integrals.
    a = u * v * dx + sum([dt * context.scale(j-1) * D[j] * inner(grad(v), grad(u)) * dx(j) for j in range(1, 4)])
    # Define linear form.
    L = U_prev * v * dx

    A = assemble(a)
    bc = DirichletBC(V, 0, context.boundaries, 1)
    bc.apply(A)
    # Define solver. Use GMRES iterative method with AMG preconditioner.
    solver = LinearSolver(mpi_comm_self(), *context.linear_solver_args)
    solver.set_operator(A)
    solver.parameters['absolute_tolerance'] = 10**-6  

    solver.parameters['maximum_iterations'] = 100  
    solver.parameters['monitor_convergence'] = True  
    solver.parameters['nonzero_initial_guess'] = False # this may be used to speed up  

    solver.parameters['relative_tolerance'] = 10**-6
    solver.parameters['report'] = True  


    while not context.should_stop():
        U_prev.assign(U)
        context.advance_time()
        bc = context.next_bc()

        # Assemble RHS and apply DirichletBC
        b = assemble(L)
        A.bcs = [bc]
        bc.apply(b)

        print ( "b.norm ", b.norm("l2"))
        print ("A * u ", (A*U.vector()).norm("l2"))
 

        # Solve linear system for this timestep
        solver.solve(U.vector(), b)

       

        context.handle_solution(U)

    return context.return_value()


class Context(object):
    def __init__(self, mesh_config, V, D, g_list):
        self.ds = mesh_config["ds"]
        self.dx = mesh_config["dx"]
        self.boundaries = mesh_config["boundaries"]
        self.V = V
        self.D = D
        self.g_list = g_list
        self.t = 0                                # Lars : start tid første tau ?
        self.ic = Function(self.V)
        self.linear_solver_args = ("gmres", "amg")

    def scale(self, i):
        return 1.0

    def should_stop(self):
        """Return True if the solve loop should stop."""
        raise NotImplementedError

    def advance_time(self):
        """Advance time by one timestep"""
        raise NotImplementedError

    def handle_solution(self, U):
        """Handle the solution U at this timestep."""
        raise NotImplementedError

    def next_bc(self):
        """Return the next bc to be used in solve"""
        raise NotImplementedError

    def return_value(self):
        return None


def gradient(mesh_config, V, D, g_list, tau, obs_file, alpha=0.0, beta=0.0):
    class GradientContext(Context):
        def __init__(self, mesh_config, V, D, g_list, tau, obs_file, alpha=0.0, beta=0.0):
            super(GradientContext, self).__init__(mesh_config, V, D, g_list)
            self.tau = tau
            self.next_tau = 0
            self.g = None
            self.current_g_index = 0
            self.J = 0.0
            self.d = Function(self.V)
            self.alpha = alpha
            self.beta = beta
            self.obs_file = HDF5File(mpi_comm_world(), obs_file, 'r')
            self.dt = tau[-1]/float(len(g_list))
            self.obs_file.read(self.ic, "0")

        def should_stop(self):
            return not self.next_tau < len(self.tau)

        def advance_time(self):
            self.t += self.dt
            self.g = self.g_list[self.current_g_index]
            self.current_g_index += 1

        def handle_solution(self, U):
            if abs(self.t - self.tau[self.next_tau]) < abs(self.t + self.dt - self.tau[self.next_tau]): # Lars : Enklere?
                # If t is closest to next observation then compute misfit.
                self.obs_file.read(self.d, str(self.tau[self.next_tau]))  # Read observation
                self.J += assemble((U - self.d) ** 2 * self.dx)

                # Move on to next observation
                self.next_tau += 1

            # Choose time integral weights
            if self.t <= self.dt or self.next_tau >= len(self.tau):
                # If endpoints use 0.5 weight
                weight = 0.5
            else:
                # Otherwise 1.0 weight
                weight = 1.0

            # Add regularisation
            self.J += 1 / 2 * weight * self.dt * assemble(self.g ** 2 * self.ds(1)) * self.alpha
            if self.current_g_index > 1:
                g_prev = self.g_list[self.current_g_index - 1]
                self.J += 1 / 2 * weight * self.dt * assemble(((self.g - g_prev) / self.dt) ** 2 * self.ds(1)) * self.beta

        def next_bc(self):
            return DirichletBC(self.V, self.g, self.boundaries, 1)

        def return_value(self):
            self.obs_file.close()
            return self.J

    context = GradientContext(mesh_config, V, D, g_list, tau, obs_file, alpha, beta)
    J = forward_problem(context)
    ctrls = ([Control(D[i]) for i in range(1, 4)]
             + [Control(g_i) for g_i in g_list])
    Jhat = ReducedFunctional(J, ctrls)
    Jhat.optimize()
    dJdm = Jhat.derivative()
    set_working_tape(Tape())
    return dJdm


def functional(mesh_config, V, D, g_list, tau, obs_file, alpha=0.0, beta=0.0, gradient=None):
    class FunctionalContext(Context):
        def __init__(self, mesh_config, V, D, g_list, tau, obs_file, alpha=0.0, beta=0.0, gradient=None):
            super(FunctionalContext, self).__init__(mesh_config, V, D, g_list)
            self.tau = tau
            self.next_tau = 1 # Lars: la tau[0] være inital betingelser
            self.g = None
            self.current_g_index = 0
            self.J = 0.0
            self.d = Function(self.V)
            self.alpha = alpha
            self.beta = beta
            self.obs_file = HDF5File(mpi_comm_world(), obs_file, 'r')
            self.t = tau[0]
            self.dt =( tau[-1] - tau[0] )/float(len(g_list)) # Lars : Endring for en mer generalisert metode (tau[0] = 0 )
            self.obs_file.read(self.ic, "%0.2f"%(self.t) )
            self.gradient = [1.0, 1.0, 1.0]

        def scale(self, i):
            if self.gradient is not None:
                return abs(float(self.gradient[i]))
            return 1.0

        def should_stop(self):
            return not self.next_tau < len(self.tau) -self.dt/2

        def advance_time(self):
            self.t += self.dt
            self.obs_file.read(self.g_list[self.current_g_index], "%0.2f"%(self.tau[self.next_tau])) 
            self.g = self.g_list[self.current_g_index]
            self.current_g_index += 1

        def handle_solution(self, U):
            if abs(self.t - self.tau[self.next_tau]) < abs(self.t + self.dt - self.tau[self.next_tau]): #Lars :  Enklere ? 
                # If t is closest to next observation then compute misfit.
                self.obs_file.read(self.d, "%0.2f"%(self.tau[self.next_tau]))  # Read observation
                self.J += assemble((U - self.d) ** 2 * self.dx)

                # Move on to next observation
                self.next_tau += 1

            # Choose time integral weights
            if self.t <= self.dt or self.next_tau >= len(self.tau):
                # If endpoints use 0.5 weight
                weight = 0.5
            else:
                # Otherwise 1.0 weight
                weight = 1.0

            # Add regularisation
            self.J += 1 / 2 * weight * self.dt * assemble(self.g ** 2 * self.ds(1) + grad(self.g) ** 2 * self.ds) * self.alpha
            if self.current_g_index > 1:
                g_prev = self.g_list[self.current_g_index - 2]
                self.J += 1 / 2 * weight * self.dt * assemble(((self.g - g_prev) / self.dt) ** 2 * self.ds) * self.beta

        def next_bc(self):           
            return DirichletBC(self.V, self.g, self.boundaries, 1)

        def return_value(self):
            self.obs_file.close()
            return self.J

    context = FunctionalContext(mesh_config, V, D, g_list, tau, obs_file, alpha, beta, gradient)
    return forward_problem(context)


def generate_observations(mesh_config, V, D, g_list, ic, tau, output_file):
    class ObservationsContext(Context):
        def __init__(self, mesh_config, V, D, g_list, ic, tau, obs_file):
            super(ObservationsContext, self).__init__(mesh_config, V, D, g_list)
            self.tau = tau
            self.next_tau = 0
            self.g = None
            self.current_g_index = 0
            self.J = 0.0
            self.obs_file = HDF5File(mpi_comm_world(), obs_file, 'w')
            self.dt = (tau[-1]-tau[0])/float(len(g_list))
            self.ic = ic
            self.obs_file.write(ic, "0.0")
            self.pvd = File("init.pvd") 
            #self.Exp = Expression("A+B*t-C*t*t", A=0.3, B=0.167,C=0.007 , t=0.0 , degree=1)


        def should_stop(self):
            return not self.t < self.tau[-1] - self.dt/2

        def advance_time(self):
            self.t += self.dt
            #self.Exp.t = self.t
            #self.g_list[self.current_g_index].interpolate(self.Exp)
            self.g = self.g_list[self.current_g_index]
            self.current_g_index += 1
           


        def handle_solution(self, U):
            self.obs_file.write(U, "%0.2f"%self.t ) # Write observation
            self.pvd << (U,self.t)
        def next_bc(self):
            return DirichletBC(self.V, self.g, self.boundaries, 1)

        def return_value(self):
            self.obs_file.close()

    context = ObservationsContext(mesh_config, V, D, g_list, ic, tau, output_file)
    return forward_problem(context)


def save_control_values(m, results_folder_save):
    h5file = HDF5File(mpi_comm_world(), "results/{}/opt_ctrls.xdmf".format(results_folder_save), 'w')
    if mpi_comm_self().rank == 0:
        myfile = open("results/{}/opt_consts.txt".format(results_folder_save), "w")
    for i, mi in enumerate(m):
        if isinstance(mi, Constant):
            c_list = mi.values()
            if mpi_comm_self().rank == 0:
                myfile.write("{}\n".format(str(float(c_list))))
        else:
            h5file.write(mi, str(i))
    h5file.close()
    if mpi_comm_self().rank == 0:
        myfile.close()


def load_control_values(k, V, results_folder_load):
    m = []
    myfile = open("results/{}/opt_consts.txt".format(results_folder_load), "r")
    lines = myfile.readlines()
    for i in lines:
        m.append(Constant(float(i)))
    myfile.close()
    l = len(m)

    h5file = HDF5File(mpi_comm_world(), "results/{}/opt_ctrls.xdmf".format(results_folder_load), 'r')
    for i in range(k):
        mi = Function(V)
        h5file.read(mi, str(i + l))
        m.append(mi)
    h5file.close()
    return m



