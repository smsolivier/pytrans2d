#!/usr/bin/env python3

import numpy as np

from trans2d import * 
import pytest 

def h1diffusion(Ne, p):
	mesh = RectMesh(Ne, Ne)
	space = H1Space(mesh, LagrangeBasis, p)
	print('Nu = {}'.format(space.Nu))

	K = Assemble(space, DiffusionIntegrator, lambda x: 1, p)
	M = Assemble(space, MassIntegrator, lambda x: .1, p)
	f = AssembleRHS(space, DomainIntegrator, lambda x: (2*np.pi**2 + .1)*np.sin(np.pi*x[0])*np.sin(np.pi*x[1]), p)
	A = K+M

	A = A.tolil()
	A[space.bnodes,:] = 0 
	A[space.bnodes,space.bnodes] = 1
	f[space.bnodes] = 0 

	T = GridFunction(space)
	T.data = spla.spsolve(A.tocsc(), f)

	err = T.L2Error(lambda x: np.sin(np.pi*x[0])*np.sin(np.pi*x[1]), p)
	return err 

Ne = 5
@pytest.mark.parametrize('p', [1, 2, 3, 4])
@pytest.mark.parametrize('solver', [h1diffusion])
def test_ooa(solver, p):
	E1 = solver(Ne, p)
	E2 = solver(2*Ne, p)
	ooa = np.log2(E1/E2)
	if (abs(p+1-ooa)>.15):
		print('{:.3e}, {:.3e}'.format(E1, E2))
	assert(abs(p+1-ooa)<.15)
