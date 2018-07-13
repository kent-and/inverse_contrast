from dolfin import *
from fenics_adjoint import *
from numpy.random import rand
from collections import OrderedDict

class DiffCoeff(Expression):
    def eval_cell(self, values, x, ufl_cell):
        if  self.mf[ufl_cell.index]==1:
            values[0] = self.Dc
        elif self.mf[ufl_cell.index]==2:
            values[0] = self.Dg
        else: 
            values[0] = self.Dw


class DerivDcDiffCoeff(Expression):
    def eval_cell(self, values, x,ufl_cell):
        if   self.mf[ufl_cell.index]==1:
            values[0] = 1.0
        elif self.mf[ufl_cell.index]==2:
            values[0] = 0.0 
        else: 
            values[0] = 0.0


class DerivDgDiffCoeff(Expression):
    def eval_cell(self, values, x,ufl_cell):
        if   self.mf[ufl_cell.index]==1:
            values[0] = 0.0
        elif self.mf[ufl_cell.index]==2:
            values[0] = 1.0
        else: 
            values[0] = 0.0


class DerivDwDiffCoeff(Expression):
    def eval_cell(self, values, x,ufl_cell):
        if   self.mf[ufl_cell.index]==1:
            values[0] = 0.0
        elif self.mf[ufl_cell.index]==2:
            values[0] = 0.0
        else: 
            values[0] = 1.0



def solving(alpha, beta,write_observations =False, noise_amplitude=0.0 ):
    mesh = Mesh("coarse_mesh.xml")
    domains = MeshFunction('size_t',mesh,"coarse_sub_corrected.xml")
    boundaries = FacetFunction("size_t", mesh )
    boundaries.set_all(0)	
    D = mesh.topology().dim()


    mesh.init(D-1,D) # Build connectivity between facets and cells
    for f in facets(mesh):
        if len(f.entities(D))==1 : 
           boundaries.array()[f.index()]=domains[int(f.entities(D))]


    V = FunctionSpace(mesh, "Lagrange", 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    U = Function(V)
    U_prev = Function(V)
    U_noise = Function(V)
    g = OrderedDict()

    T = 8.0
    t = 0
    dt_val = 2.0
    dt = Constant(dt_val)

    D = DiffCoeff(degree=1)
    D.mf= domains
    D.Dc = Constant(1000)
    D.Dg = Constant(1)
    D.Dw = Constant(2)
    D.user_defined_derivatives = {D.Dc: DerivDcDiffCoeff(degree=1),
                              D.Dg: DerivDgDiffCoeff(degree=1),
                              D.Dw: DerivDwDiffCoeff(degree=1), }

    ctrls = [Control(D.Dc), Control(D.Dg), Control(D.Dw)]
    D_proj = project(D, V)

    a = u * v * dx + dt * D * inner(grad(u), grad(v)) * dx
    L = U_prev * v * dx

    times=[]

    DCbc = Expression("A+B*t", A=0.3, B=1.0 , t=0.0 , degree=1)

    if write_observations:
       observations = HDF5File(mpi_comm_world(), "U.xdmf", "w")
       #bc = DirichletBC(V,Constant(1.0),boundaries,1)
       bc = DirichletBC(V,DCbc,boundaries,1)
    else:
       observations = HDF5File(mpi_comm_world(), "U.xdmf", "r")
       obs_func = Function(V)
       g[t] = Function(V, name="boundary", annotate=True) 
       bc = DirichletBC(V,g[t],"on_boundary",annotate = True)

    J = 0
   
    while t <= T:

       solve(a == L, U, bc)

       U_prev.assign(U)
    
       t += dt_val
       print("Time ", t)
       times.append(t)

       if write_observations:
          U_noise .vector()[:] = U.vector()[:] #+ 
          observations.write(U_noise, str(t))
          DCbc.t=t
          bc = DirichletBC(V,DCbc,boundaries,1)
          #bc = DirichletBC(V,Constant(1.0),boundaries,1)

       else:
          g[t]= Function(V, name="boundary", annotate=True) 
          bc = DirichletBC(V,g[t],"on_boundary",annotate = True)
          try :
              observations.read(obs_func, str(t))
              obs_func.vector()[:]+=  noise_amplitude*rand(U.vector().size()) 
              J += assemble((U - obs_func ) ** 2 * dx)
          except: 
              print("Error")

    if write_observations:
       return 0

    alpha=Constant(alpha)
    beta =Constant(beta)
   
    regularisation = alpha/2*sum([dt/2*(gb+ga)**2*ds for gb, ga in zip(list(g.values())[1:], list(g.values())[:-1])]) 
    smoothness = beta/2*sum([1/dt*(gb-ga)**2*ds for gb, ga in zip(list(g.values())[1:], list(g.values())[:-1])])

    J += assemble(regularisation)
    J += assemble(smoothness)
    print("Functional value:", J)
    print type(J) 
    print("Computing gradient")

    ctrls+=[ Control(c) for c in g.values()]

    Jhat = ReducedFunctional(J, ctrls)



    m = minimize(Jhat, options={"maxiter": 50})

    j = Jhat(m)
    optDC= File("optDC.pvd")
    print("Functional value after optimization: {}".format(j))
    print(m[0].values(),m[1].values(),m[2].values())      
    for no,k in enumerate(times):
       print no,k
       observations.read(obs_func, str(k))
       be = assemble( (m[no+3]-obs_func)**2*ds )
       optDC << (m[no+3],k)
       print be 
        

if __name__=='__main__':
	import argparse
        import sys
	parser = argparse.ArgumentParser(prog='mri-contrast.py')
	parser.add_argument('--alpha', default=1.0, type=float)
 
	parser.add_argument('--beta',  default=1.0,type=float)
        parser.add_argument('--noise',  default=0.0)
	Z = parser.parse_args()
        print Z

        solving(Z.alpha, Z.beta , True )
        solving(Z.alpha, Z.beta , False)
