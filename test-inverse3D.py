from dolfin import *
from fenics_adjoint import *
from numpy.random import rand


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




mesh = Mesh("coarse_mesh.xml")
domains = MeshFunction('size_t',mesh,"coarse_sub_corrected.xml")

boundaries = FacetFunction("size_t", mesh )
boundaries.set_all(0)	
D = mesh.topology().dim()
mesh.init(D-1,D) # Build connectivity between facets and cells
for f in facets(mesh):
    if len(f.entities(D))==1 : 
       boundaries.array()[f.index()]=	domains[int(f.entities(D))]

#plot(boundaries, interactive=True)

V = FunctionSpace(mesh, "Lagrange", 1)
u = TrialFunction(V)
v = TestFunction(V)

U = Function(V)
U_prev = Function(V)
U_noise = Function(V)
g = Function(V, name="Control")

T = 1.0
t = 0
dt_val = 0.1
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



# plot(c_proj)

a = u * v * dx + dt * D * inner(grad(u), grad(v)) * dx
L = U_prev * v * dx
# A = assemble(a)

bc = DirichletBC(V,Constant(1.0),boundaries,1)
#bc = DirichletBC(V,g,"on_boundary")

write_observations = False
if write_observations:
    observations = HDF5File(mpi_comm_world(), "U.xdmf", "w")
else:
    observations = HDF5File(mpi_comm_world(), "U.xdmf", "r")
    obs_func = Function(V)

J = 0
while t <= T:

    solve(a == L, U, bc)

    U_prev.assign(U)

    # plot(U)


    t += dt_val
    print("Time ", t)

    # Write observation
    if write_observations:
        U_noise .vector()[:] = U.vector()[:] #+ 0.05*rand(U.vector().size()) 
        observations.write(U_noise, str(t))
    else:
        try :
            observations.read(obs_func, str(t))
            J += assemble((U - obs_func) ** 2 * dx)
        except: 
             print "Error"

if write_observations:
    exit()

print("Functional value:", J)
print type(J) 
print("Computing gradient")

Jhat = ReducedFunctional(J, ctrls)

# exit()
m = [Constant(1000), Constant(2), Constant(2)]
U.vector()[:] = 0
U_prev.vector()[:] = 0


j = Jhat(m)
print("Functional value at the start: {}".format(j))

m = minimize(Jhat,'CG', options={"maxiter": 50})

j = Jhat(m)
print("Functional value after optimization: {}".format(j))



#for i in range(10):
    # Evaluate forward model at new control values
#    j = Jhat(m)
#    print("Functional value at iteration {}: {}".format(i, j))

    # Compute gradient
#    dJdm = compute_gradient(J, ctrls)

    # Update control values:
#    alpha = 0.1
#    m = [Constant(m[0] - alpha / dJdm[0]),
#         Constant(m[1] - alpha / dJdm[1]),
#         Constant(m[2] - alpha / dJdm[2])]
#    print([float(mm) for mm in m])

#exit()
print("Running Taylor test")

from IPython import embed;

embed()
h = [Constant(10), Constant(1), Constant(1)]
taylor_test_multiple(Jhat, m, h)
