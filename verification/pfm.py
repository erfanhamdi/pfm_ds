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

def psi_pos_m(u):
    return 0.5*lmbda*(bracket_pos(ufl.tr(epsilon(u)))**2) + mu*(ufl.inner(epsilon_p, epsilon_p))

def psi_neg_m(u):
    return 0.5*lmbda*(bracket_neg(ufl.tr(epsilon(u)))**2) + mu*(ufl.inner(epsilon_n, epsilon_n))
