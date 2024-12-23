import ufl
import numpy as np
import matplotlib.pyplot as plt
import ufl.constant

from dolfinx import mesh, fem, plot, io, default_scalar_type
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc, create_vector, create_matrix
from mpi4py import MPI
from petsc4py import PETSc
import pyvista as pv
from pathlib import Path

import os
import time

from utils import plotter_func, plot_force_disp, distance_points_to_segment

ksp = PETSc.KSP.Type.GMRES
pc = PETSc.PC.Type.HYPRE

crack_id = 2
crack_id = int(crack_id)-1
mesh_size = 100
crack_pattern_id = 2
start_time = time.time()

seed = crack_id
out_file = f"./results/{mesh_size}_{seed}_ufl"
# create directory
Path(out_file).mkdir(parents=True, exist_ok=True)
crack_pattern = f"/projectnb/lejlab2/erfan/dataset_phase1/data/b4/cracks/{crack_pattern_id}.npy"

print(f"seed = {seed}, mesh = {mesh_size}, crack = {crack_pattern}, out_file = {out_file}")

results_folder = Path(out_file)
results_folder.mkdir(exist_ok=True, parents=True)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0, 0]), np.array([2, 2])], [mesh_size, mesh_size], cell_type=mesh.CellType.quadrilateral)

G_c_ = fem.Constant(domain, 1.0)
l_0_ = fem.Constant(domain, 0.01)
E = fem.Constant(domain, 1000.0e3)
nu = fem.Constant(domain, 0.3)
mu = E/(2*(1+nu))
lmbda = E*nu/((1+nu)*(1-2*nu))

V = fem.functionspace(domain, ("Lagrange", 1,))
W = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
VV = fem.functionspace(domain, ("DG", 0,))

u, v = ufl.TrialFunction(W), ufl.TestFunction(W)
p, q = ufl.TrialFunction(V), ufl.TestFunction(V)

u_new, u_old = fem.Function(W), fem.Function(W)
p_new, H_old, p_old = fem.Function(V), fem.Function(VV), fem.Function(V)
H_init_ = fem.Function(V)

tdim = domain.topology.dim
fdim = tdim - 1

def top_boundary(x):
    return np.isclose(x[1], 2)

def left_boundary(x):
    return np.isclose(x[0], 0)

def right_boundary(x):
    return np.isclose(x[0], 2)

def bottom_boundary(x):
    return np.isclose(x[1], 0)

top_facet = mesh.locate_entities_boundary(domain, fdim, top_boundary)
top_marker = 1
top_marked_facets = np.full_like(top_facet, top_marker)

bot_facet = mesh.locate_entities_boundary(domain, fdim, bottom_boundary)
bot_marker = 2
bot_marked_facets = np.full_like(bot_facet, bot_marker)

right_facet = mesh.locate_entities_boundary(domain, fdim, right_boundary)
right_marker = 3
right_marked_facets = np.full_like(right_facet, right_marker)

left_facet = mesh.locate_entities_boundary(domain, fdim, left_boundary)
left_marker = 4
left_marked_facets = np.full_like(left_facet, left_marker)

marked_facets = np.hstack([top_facet, bot_facet, right_facet, left_facet])
marked_values = np.hstack([np.full_like(top_facet, 1), np.full_like(bot_facet, 2), np.full_like(right_facet, 3), np.full_like(left_facet, 4)])
sorted_facets = np.argsort(marked_facets)
facet_tag = mesh.meshtags(domain, fdim, marked_facets[sorted_facets], marked_values[sorted_facets])

top_x_dofs = fem.locate_dofs_topological(W.sub(0), fdim, top_facet)
top_y_dofs = fem.locate_dofs_topological(W.sub(1), fdim, top_facet)

bot_x_dofs = fem.locate_dofs_topological(W.sub(0), fdim, bot_facet)
bot_y_dofs = fem.locate_dofs_topological(W.sub(1), fdim, bot_facet)

right_x_dofs = fem.locate_dofs_topological(W.sub(0), fdim, right_facet)
right_y_dofs = fem.locate_dofs_topological(W.sub(1), fdim, right_facet)

left_x_dofs = fem.locate_dofs_topological(W.sub(0), fdim, left_facet)
left_y_dofs = fem.locate_dofs_topological(W.sub(1), fdim, left_facet)

u_bc_top = fem.Constant(domain, default_scalar_type(0.0))
u_bc_right = fem.Constant(domain, default_scalar_type(0.0))

bc_bot_y = fem.dirichletbc(default_scalar_type(0.0), bot_y_dofs, W.sub(1))
bc_left_x = fem.dirichletbc(default_scalar_type(0.0), left_x_dofs, W.sub(0))

bc_top = fem.dirichletbc(u_bc_top, top_y_dofs, W.sub(1))
bc_right = fem.dirichletbc(u_bc_top, right_x_dofs, W.sub(0))

bc = [bc_bot_y, bc_top, bc_right, bc_left_x]

ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag)
dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 2})

def epsilon(u):
    return ufl.sym(ufl.grad(u))

def sigma(u):
    return lmbda*ufl.tr(epsilon(u))*ufl.Identity(2) + 2.0*mu*epsilon(u)

def bracket_pos(u):
    return 0.5*(u + np.abs(u))

def bracket_neg(u):
    return 0.5*(u - np.abs(u))
##################################################################
A = ufl.variable(epsilon(u_new))
I1 = ufl.tr(A)
delta = (A[0, 0] - A[1, 1])**2 + 4 * A[0, 1] * A[1, 0]
delta += 3.0e-16 ** 2
eigval_1 = (I1 - ufl.sqrt(delta)) / 2
eigval_2 = (I1 + ufl.sqrt(delta)) / 2
eigvec_1 = ufl.diff(eigval_1, A).T
eigvec_2 = ufl.diff(eigval_2, A).T
epsilon_p = 0.5 * (eigval_1 + abs(eigval_1)) * eigvec_1 + 0.5 * (eigval_2 + abs(eigval_2)) * eigvec_2
epsilon_n = 0.5 * (eigval_1 - abs(eigval_1)) * eigvec_1 + 0.5 * (eigval_2 - abs(eigval_2)) * eigvec_2

def psi_pos_m(u):
    return 0.5*lmbda*(bracket_pos(ufl.tr(epsilon(u)))**2) + mu*(ufl.inner(epsilon_p, epsilon_p))

def psi_neg_m(u):
    return 0.5*lmbda*(bracket_neg(ufl.tr(epsilon(u)))**2) + mu*(ufl.inner(epsilon_n, epsilon_n))

def H(u_new, H_old):
    return ufl.conditional(ufl.gt(psi_pos_m(u_new), H_old), psi_pos_m(u_new), H_old)

def H_init(dist_list, l_0, G_c):
    distances = np.array(dist_list)
    distances = np.min(distances, axis=0)
    mask0 = distances <= l_0.value/2
    H = np.zeros_like(distances)
    phi_c = 0.999
    H[mask0] = ((phi_c/(1-phi_c))*G_c.value/(2*l_0.value))*(1-(2*distances[mask0]/l_0.value))
    return H

cracks = np.load(crack_pattern)
A_ = cracks[:, 0, :]
B_ = cracks[:, 1, :]
points = domain.geometry.x[:, :2]
dist_list = []

for idx in range(len(A_)):
    distances = distance_points_to_segment(points, A_[idx][0], A_[idx][1], B_[idx][0], B_[idx][1])
    dist_list.append(distances)

H_init_.x.array[:] = H_init(dist_list, l_0_, G_c_)
H_old.interpolate(H_init_)

T = fem.Constant(domain, default_scalar_type((0, 0)))
E_du = ((1.0-p_new)**2)*ufl.inner(ufl.grad(v),sigma(u))*dx + ufl.dot(T, v) * ds
a_u = fem.form(ufl.lhs(E_du))
L_u = fem.form(ufl.rhs(E_du))
A_u = create_matrix(a_u)
b_u = create_vector(L_u)

solver_u = PETSc.KSP().create(domain.comm)
solver_u.setOperators(A_u)
solver_u.setType(ksp)
solver_u.getPC().setType(pc)

E_phi = (((l_0_**2) * ufl.dot(ufl.grad(p), ufl.grad(q))) + ((2*l_0_/G_c_) * H(u_new, H_old) +1 ) * p * q )* dx - (2*l_0_/G_c_) * H(u_new, H_old) * q * dx
a_phi = fem.form(ufl.lhs(E_phi))
L_phi = fem.form(ufl.rhs(E_phi))
A_phi = create_matrix(a_phi)
b_phi = create_vector(L_phi)

solver_phi = PETSc.KSP().create(domain.comm)
solver_phi.setOperators(A_phi)
solver_phi.setType(ksp)
solver_phi.getPC().setType(pc)

num_steps = 6000
t_ = 0
delta_T1 = 1e-6

B_bot_y_new = []
B_left_x_new = []

du = ufl.TrialFunction(W)
u_l2_error = fem.form(ufl.dot(u_new - u_old, u_new - u_old)*dx)
p_l2_error = fem.form(ufl.dot(p_new - p_old, p_new - p_old)*dx)

R_bot_form_y = fem.form(((1.0-p_new)**2)*sigma(u_new)[1, 1] * ds(2))
R_left_form_x = fem.form(((1.0-p_new)**2)*sigma(u_new)[0, 0] * ds(4))

############################ new rxn force calculation ###############################################
residual = ufl.action(ufl.lhs(E_du), u_new) - ufl.rhs(E_du)

v_reac = fem.Function(W)
virtual_work_form = fem.form(ufl.action(residual, v_reac))

left_dofs = fem.locate_dofs_geometrical(W, left_boundary)
bot_dofs = fem.locate_dofs_geometrical(W, bottom_boundary)
top_dofs = fem.locate_dofs_geometrical(W, top_boundary)
right_dofs = fem.locate_dofs_geometrical(W, right_boundary)

u_bc_left = fem.Function(W)
u_bc_bot = fem.Function(W)
u_bc_top_rxn = fem.Function(W)
u_bc_right_rxn = fem.Function(W)

bc_left_rxn = fem.dirichletbc(u_bc_left, left_dofs)
bc_bot_rxn = fem.dirichletbc(u_bc_bot, bot_dofs)
bc_top_rxn = fem.dirichletbc(u_bc_top_rxn, top_dofs)
bc_right_rxn = fem.dirichletbc(u_bc_right_rxn, right_dofs)

bc_rxn = [bc_left_rxn, bc_bot_rxn]

def one(x):
    values = np.zeros((1, x.shape[1]))
    values[0] = 1.0
    return values

u_bc_left.sub(0).interpolate(one)
u_bc_bot.sub(1).interpolate(one)

############################ new rxn force calculation ###############################################
delta_T = delta_T1
for i in range(num_steps+1):
    step_time = time.time()
    print(f"Step {i}/{num_steps}")
    t_ += delta_T
    u_bc_top.value = t_
    error_total = 1
    error_tol = 1e-5
    flag = 1
    print(f"rank {rank}: t_ = {t_}")
    staggered_iter = 0
    while flag:
        staggered_iter +=1
        if error_total < error_tol:
            flag = 0
            break
        A_u.zeroEntries()
        assemble_matrix(A_u, a_u, bcs = bc)
        A_u.assemble()
        with b_u.localForm() as loc:
            loc.set(0)
        assemble_vector(b_u, L_u)
        apply_lifting(b_u, [a_u], [bc])
        b_u.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b_u, bc)
        solver_u.solve(b_u, u_new.vector)
        u_new.x.scatter_forward()
        
        A_phi.zeroEntries()
        assemble_matrix(A_phi, a_phi, bcs = [])
        A_phi.assemble()
        with b_phi.localForm() as loc:
            loc.set(0)
        assemble_vector(b_phi, L_phi)
        b_phi.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        solver_phi.solve(b_phi, p_new.vector)
        p_new.x.scatter_forward()

        error_u = np.sqrt(domain.comm.allreduce(fem.assemble_scalar(u_l2_error), op=MPI.SUM))
        error_p = np.sqrt(domain.comm.allreduce(fem.assemble_scalar(p_l2_error), op=MPI.SUM))
        error_total = error_u + error_p
        print(f"rank = {rank}, iter = {staggered_iter}, error_u = {error_u}, error_p = {error_p}, total = {error_total}")
        p_old.x.array[:] = p_new.x.array
        u_old.x.array[:] = u_new.x.array
        H_old.interpolate(fem.Expression(H(u_new, H_old), VV.element.interpolation_points()))
    ################################################################################
    R_bot_y = domain.comm.gather(fem.assemble_scalar(R_bot_form_y), root=0)
    R_left_x = domain.comm.gather(fem.assemble_scalar(R_left_form_x), root=0)

    v_reac.vector.set(0.0)
    v_reac.x.scatter_forward()
    fem.set_bc(v_reac.vector, [bc_left_rxn])
    R_left_x_new = domain.comm.gather(fem.assemble_scalar(virtual_work_form), root=0)

    v_reac.vector.set(0.0)
    v_reac.x.scatter_forward()
    fem.set_bc(v_reac.vector, [bc_bot_rxn])
    R_bot_y_new = domain.comm.gather(fem.assemble_scalar(virtual_work_form), root=0)

    if domain.comm.rank == 0:
        B_bot_y_new.append([np.sum(R_bot_y_new), t_])
        B_left_x_new.append([np.sum(R_left_x_new), t_])

    if i%100 == 0:
        out_file_name = f"{out_file}/p_mpi_{i}.xdmf"
        out_file_name_u = f"{out_file}/u_mpi_{i}.xdmf"

        with io.XDMFFile(domain.comm, out_file_name, "w") as xdmf:
            xdmf.write_mesh(domain)
            xdmf.write_function(p_new)
        with io.XDMFFile(MPI.COMM_WORLD, out_file_name_u, "w") as xdmf:
            xdmf.write_mesh(domain)
            xdmf.write_function(u_new)

        plotter_func(p_new, dim=1, mesh = domain, title=f"{out_file}/p_mpi_{i}")
        if rank == 0:
            plot_force_disp(B_bot_y_new, "bot_y", out_file)
            plot_force_disp(B_left_x_new, "left_x", out_file)
        
end_time = time.time()
print(f"total time = {start_time - end_time}")