#!/usr/bin/env python3

import numpy as np 
import pytest

from trans2d import *

def test_amg():
	Ne = 10 
	p = 1 
	mesh = RectMesh(Ne, Ne)
	space = H1Space(mesh, LagrangeBasis, p)

	K = Assemble(space, DiffusionIntegrator, lambda x: 1, 2*p)
	M = Assemble(space, MassIntegrator, lambda x: .1, 2*p+1)
	f = AssembleRHS(space, DomainIntegrator, lambda x: 1, 2*p+1)
	A = K+M

	A = A.tolil()
	A[space.bnodes,:] = 0 
	A[space.bnodes,space.bnodes] = 1
	f[space.bnodes] = 0 

	T = GridFunction(space)
	solver = AMGSolver(1e-12, 20, 1, False)
	T.data = solver.Solve(A, A, f)

	res = np.linalg.norm(A*T.data - f)
	assert(res<1e-12)

def test_blockldu():
	Ne = 10 
	p = 1 
	mesh = RectMesh(Ne, Ne)
	phi_space = L2Space(mesh, LegendreBasis, p)
	J_space = H1Space(mesh, LobattoBasis, p+1, 2)
	eps = 1e-2
	sigma_t = lambda x: 1/eps 
	sigma_a = lambda x: eps
	Q = lambda x: 1
	qorder = 2*p+2
	Mt = 3*Assemble(J_space, VectorMassIntegrator, sigma_t, qorder)
	Mtl = 3*Assemble(J_space, VectorMassIntegratorRowSum, sigma_t, qorder)
	Ma = Assemble(phi_space, MassIntegrator, sigma_a, qorder)
	D = MixAssemble(phi_space, J_space, MixDivIntegrator, 1, qorder)
	f = AssembleRHS(phi_space, DomainIntegrator, Q, qorder)
	A = sp.bmat([[Mt, -D.transpose()], [D, Ma]])
	rhs = np.concatenate((np.zeros(J_space.Nu), f))

	solver = BlockLDU(1e-10, 50, 1, False)
	Ainv = sp.diags(1/Mtl.diagonal())
	S = Ma + D*Ainv*D.transpose()
	x = solver.Solve(Mt, -D.transpose(), D, Ma, Ainv, S, A, rhs)
	res = np.linalg.norm(A*x - rhs)
	assert(res<1e-10)

@pytest.mark.parametrize('solver', [Jacobi, GaussSeidel, SymGaussSeidel])
def test_smooth(solver):
	N = 10 
	A = sp.lil_matrix((N,N))
	A[0,0] = 1 
	A[-1,-1] = 1 
	for i in range(1,N-1):
		A[i,i] = 2 
		A[i,i-1] = A[i,i+1] = -1 

	rhs = np.ones(N)
	rhs[0] = rhs[-1] = 0 

	s = solver(1e-10, 10000, False)
	x = s.Solve(A.tocsc(), rhs)
	assert(s.IsConverged())
	r = np.linalg.norm(A*x - rhs)
	assert(r<1e-9)