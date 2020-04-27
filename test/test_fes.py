#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pytest

from trans2d import * 

mesh = RectMesh(3, 3)
space = H1Space(mesh, LagrangeBasis, 2, 2)
l2 = L2Space(mesh, LagrangeBasis, 2, 2)

def test_fes():
	assert(space.Nu==98)

def test_l2():
	assert(l2.Nu==9*9*2)

def test_lorefine():
	p = 3
	space = H1Space(mesh, LobattoBasis, p, 2)
	l2 = L2Space(mesh, LegendreBasis, p-1)
	lospace = space.LORefine() 
	assert(lospace.Nu==space.Nu)
	assert(lospace.Ne==space.Ne*p**2) 
	lol2 = L2Space(lospace.mesh, LegendreBasis, 0)
	assert(lol2.Nu==l2.Nu)

	lomesh = lospace.mesh 
	s = 0
	for e in range(p**2):
		trans = lomesh.trans[e] 
		s += trans.Area() 

	assert(s == pytest.approx(mesh.trans[0].Area()))
	assert(np.max(space.dofs)==np.max(lospace.dofs))

def lorefine_diffusion_solve(Ne, p):
	mesh = RectMesh(Ne, Ne)
	ho_space = H1Space(mesh, LobattoBasis, p, 2)
	J_space = ho_space.LORefine()
	mesh = J_space.mesh 
	phi_space = L2Space(mesh, LegendreBasis, 0)
	eps = 1e-2
	sigma_t = lambda x: 1/eps 
	sigma_a = lambda x: eps
	phi_ex = lambda x: np.sin(np.pi*x[0])*np.sin(np.pi*x[1])
	Jex = lambda x: [-np.pi/3/sigma_t(x)*np.cos(np.pi*x[0])*np.sin(np.pi*x[1]), 
		-np.pi/3/sigma_t(x)*np.sin(np.pi*x[0])*np.cos(np.pi*x[1])]
	Q = lambda x: (2*np.pi**2/3/sigma_t(x) + sigma_a(x))*phi_ex(x) 
	qorder = 2*p+2
	Mt = -3*Assemble(J_space, VectorMassIntegrator, sigma_t, qorder)
	Ma = Assemble(phi_space, MassIntegrator, sigma_a, qorder)
	D = MixAssemble(phi_space, J_space, MixDivIntegrator, 1, qorder)
	f = AssembleRHS(phi_space, DomainIntegrator, Q, qorder)
	A = sp.bmat([[Mt, D.transpose()], [D, Ma]])
	rhs = np.concatenate((np.zeros(J_space.Nu), f))
	x = spla.spsolve(A.tocsc(), rhs)
	phi = GridFunction(phi_space)
	J = GridFunction(J_space)
	phi.data = x[J_space.Nu:]
	J.data = x[:J_space.Nu]
	return phi.L2Error(phi_ex, 2*p+2), J.L2Error(Jex, 2*p+2)

def test_lor_mms():
	E1 = np.array(lorefine_diffusion_solve(5, 3))
	E2 = np.array(lorefine_diffusion_solve(10, 3))
	ooa = np.log2(E1/E2)
	assert(ooa == pytest.approx(np.array([1,1]), abs=.1))
