import ufl
from ufl import sqrt
import numpy as np
import matplotlib.pyplot as plt
import ufl.constant

from dolfinx import mesh, fem, plot, io, default_scalar_type
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc, create_vector, create_matrix
from mpi4py import MPI
from petsc4py import PETSc
import pyvista as pv
import sympy as sp
from pathlib import Path

import os
import time

import ufl
import dolfinx
from petsc4py import PETSc
from dolfinx.fem.petsc import apply_lifting, assemble_matrix, assemble_vector, set_bc


def project(e, target_func, bcs=[]):
    """Project UFL expression.

    Note
    ----
    This method solves a linear system (using KSP defaults).

    """

    # Ensure we have a mesh and attach to measure
    V = target_func.function_space
    dx = ufl.dx(V.mesh)

    # Define variational problem for projection
    w = ufl.TestFunction(V)
    v = ufl.TrialFunction(V)
    a = dolfinx.fem.form(ufl.inner(v, w) * dx)
    L = dolfinx.fem.form(ufl.inner(e, w) * dx)

    # Assemble linear system
    A = assemble_matrix(a, bcs)
    A.assemble()
    b = assemble_vector(L)
    apply_lifting(b, [a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, bcs)

    # Solve linear system
    solver = PETSc.KSP().create(A.getComm())
    solver.setType("bcgs")
    solver.getPC().setType("bjacobi")
    # solver.rtol = 1.0e-05
    solver.setOperators(A)
    solver.solve(b, target_func.x.petsc_vec)
    assert solver.reason > 0
    target_func.x.scatter_forward()

    # Destroy PETSc linear algebra objects and solver
    solver.destroy()
    A.destroy()
    b.destroy()

def invariants_principal(A):
    """Principal invariants of (real-valued) tensor A.
    https://doi.org/10.1007/978-3-7091-0174-2_3
    """
    i1 = ufl.tr(A)
    i2 = (ufl.tr(A)**2 - ufl.tr(A * A)) / 2
    i3 = ufl.det(A)
    return i1, i2, i3

def deviatoric_part(T):
    """
    Computes the deviatoric part of a given tensor.

    Parameters
    ----------
    T: Tensor
        The input tensor for which the deviatoric part needs to be computed.

    Returns
    -------
    Tensor
        The deviatoric part of the input tensor, computed as the tensor minus
        one-third of its trace times the identity tensor.

    Notes
    -----
    The deviatoric part of a tensor is defined as the part of the tensor that
    remains after subtracting the isotropic part. In 3D, for a second-order tensor
    T, the deviatoric part D is given by $D = T - (1/3)  tr(T)  I$, where tr(T) is the trace of T and I is the identity tensor.
    """
    return ufl.dev(T)

def volumetric_part(T):
    """
    Computes the volumetric part of a given tensor.

    Parameters
    ----------
    T: Tensor
        The input tensor for which the volumetric part needs to be computed.

    Returns
    -------
    Tensor
        The volumetric part of the input tensor, computed as the tensor minus
        its deviatoric part.

    Notes
    -----
    The volumetric part of a tensor is defined as the isotropic part of the tensor,
    i.e., the part that is invariant under rotations. In 3D, for a second-order tensor
    T, the volumetric part V is given by $V = T - dev(T)$ where dev(T) is the deviatoric part of T.
    """
    return T - ufl.dev(T)

def macaulay_bracket_positive(x):
    """
    Compute the Macaulay bracket (positive part) of a real number x.

    Parameters
    ----------
    x : float or numpy.ndarray
        Real number or array of real numbers.

    Returns
    -------
    float or numpy.ndarray
        Macaulay bracket of x, which is defined as 0.5 * (x + abs(x)).
    """
    return 0.5 * (x + abs(x))


def macaulay_bracket_negative(x):
    """
    Compute the Macaulay bracket (negative part) of a real number x.

    Parameters
    ----------
    x : float or numpy.ndarray
        Real number or array of real numbers.

    Returns
    -------
    float or numpy.ndarray
        Macaulay bracket of x, which is defined as 0.5 * (x - abs(x)).
    """
    return 0.5 * (x - abs(x))


def eigenstate2(A):
    """Eigenvalues and eigenprojectors of the 2x2 (real-valued) tensor A.
    Provides the spectral decomposition A = sum_{a=0}^{1} λ_a * E_a
    with (ordered) eigenvalues λ_a and their associated eigenprojectors E_a = n_a^R x n_a^L.

    Note: Tensor A must not have complex eigenvalues!
    """
    if ufl.shape(A) != (2, 2):
        raise RuntimeError(f"Tensor A of shape {
                           ufl.shape(A)} != (2, 2) is not supported!")
    #
    # slightly above 2**-(53 - 1), see https://en.wikipedia.org/wiki/IEEE_754
    eps = 3.0e-16
    #
    A = ufl.variable(A)
    #
    # --- determine eigenvalues λ0, λ1
    #
    I1, _, _ = invariants_principal(A)
    #
    Δ = (A[0, 0] - A[1, 1])**2 + 4 * A[0, 1] * A[1, 0]  # = I1**2 - 4 * I2
    # Avoid dp = 0 and disc = 0, both are known with absolute error of ~eps**2
    # Required to avoid sqrt(0) derivatives and negative square roots
    Δ += eps**2
    # sorted eigenvalues: λ0 <= λ1
    λ = (I1 - ufl.sqrt(Δ)) / 2, (I1 + ufl.sqrt(Δ)) / 2
    #
    # --- determine eigenprojectors E0, E1
    #
    E = [ufl.diff(λk, A).T for λk in λ]

    return λ, E

def spectral_positive_part(T):
    """
    Computes the spectral possitive part of a given tensor.

    Parameters
    ----------
    T: Tensor
        The input tensor for which the spectral possitive part needs to be computed.

    Returns
    -------
    Tensor
        The spectral possitive part of the input tensor.

    Notes
    -----
    The tensor is split into its positive and negative parts through spectral decomposition, $\boldsymbol \epsilon=\sum_{i=1}^{\alpha} \epsilon^i \boldsymbol n^i \otimes \boldsymbol n^i$ where $\epsilon^i$ are the principal strains, and $\boldsymbol n^i$ are the principal strains directions. The positive and negative parts of the tensor $\boldsymbol \epsilon    = \boldsymbol \epsilon^+ + \boldsymbol \epsilon^-$,  are defined as:

    * $\boldsymbol \epsilon^+: = \sum_{i=1}^{\alpha} \langle \epsilon^i \rangle^+ \boldsymbol n^i \otimes \boldsymbol n^i$,
    * $\boldsymbol \epsilon^-: = \sum_{i=1}^{\alpha} \langle \epsilon^i \rangle^- \boldsymbol n^i \otimes \boldsymbol n^i$.

    In which $\langle\rangle^{\pm}$ are the bracket operators. $\langle x \rangle^+:=\frac{x+|x|}{2}$, and $\langle x \rangle^-:=\frac{x-|x|}{2}$.

    """
    if ufl.shape(T) == (2, 2):
        eig, eig_vec = eigenstate2(T)
        T_p = macaulay_bracket_positive(eig[0]) * eig_vec[0]
        T_p += macaulay_bracket_positive(eig[1]) * eig_vec[1]

    return T_p

def spectral_negative_part(T):
    """
    Computes the spectral negative part of a given tensor.

    Parameters
    ----------
    T: Tensor
        The input tensor for which the spectral negative part needs to be computed.

    Returns
    -------
    Tensor
        The spectral negative part of the input tensor.

    Notes
    -----
    The tensor is split into its positive and negative parts through spectral decomposition, $\boldsymbol \epsilon=\sum_{i=1}^{\alpha} \epsilon^i \boldsymbol n^i \otimes \boldsymbol n^i$ where $\epsilon^i$ are the principal strains, and $\boldsymbol n^i$ are the principal strains directions. The positive and negative parts of the tensor $\boldsymbol \epsilon    = \boldsymbol \epsilon^+ + \boldsymbol \epsilon^-$,  are defined as:

    * $\boldsymbol \epsilon^+: = \sum_{i=1}^{\alpha} \langle \epsilon^i \rangle^+ \boldsymbol n^i \otimes \boldsymbol n^i$,
    * $\boldsymbol \epsilon^-: = \sum_{i=1}^{\alpha} \langle \epsilon^i \rangle^- \boldsymbol n^i \otimes \boldsymbol n^i$.

    In which $\langle\rangle^{\pm}$ are the bracket operators. $\langle x \rangle^+:=\frac{x+|x|}{2}$ and $\langle x \rangle^-:=\frac{x-|x|}{2}$.

    """
    if ufl.shape(T) == (2, 2):
        eig, eig_vec = eigenstate2(T)
        T_p = macaulay_bracket_negative(eig[0]) * eig_vec[0]
        T_p += macaulay_bracket_negative(eig[1]) * eig_vec[1]
    else:
        eig, eig_vec = eigenstate3(T)
        T_p = macaulay_bracket_negative(eig[0]) * eig_vec[0]
        T_p += macaulay_bracket_negative(eig[1]) * eig_vec[1]
        T_p += macaulay_bracket_negative(eig[2]) * eig_vec[2]
    return T_p    