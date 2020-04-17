#!/usr/bin/env python3

import numpy as np

from trans2d import * 
import pytest 

def TransportMMS(alpha, beta, gamma, delta, eta, sigt, sigs):
	f = lambda x: np.sin(np.pi*x[0])*np.sin(np.pi*x[1]) + delta 
	df = lambda x: np.array([np.pi*np.cos(np.pi*x[0])*np.sin(np.pi*x[1]), 
		np.pi*np.sin(np.pi*x[0])*np.cos(np.pi*x[1])])
	g = lambda x: np.sin(2*np.pi*x[0])*np.sin(2*np.pi*x[1])
	dg = lambda x: np.array([2*np.pi*np.cos(2*np.pi*x[0])*np.sin(2*np.pi*x[1]), 
		2*np.pi*np.sin(2*np.pi*x[0])*np.cos(2*np.pi*x[1])])
	h = lambda x: x[0]*(1-x[0])*x[1]*(1-x[1])
	dh = lambda x: np.array([(1-2*x[0])*x[1]*(1-x[1]), x[0]*(1-x[0])*(1-2*x[1])])
	psi_ex = lambda x,Omega: 1/4/np.pi*(
		alpha*f(x)
		+ beta*(Omega[0]+Omega[1])*g(x) 
		+ gamma*(Omega[0]**2)*h(x))
	gpsi = lambda x, Omega: 1/4/np.pi*(alpha*df(x) 
		+ beta*(Omega[0]+Omega[1])*dg(x) 
		+ gamma*(Omega[0]**2)*dh(x))
	phi = lambda x: alpha*f(x) + 1/3*gamma*h(x) 
	Q = lambda x, Omega: np.dot(Omega, gpsi(x,Omega)) + sigt(x)*psi_ex(x,Omega) - sigs(x)/4/np.pi*phi(x)
	return psi_ex, phi, Q 

def h1diffusion(Ne, p):
	mesh = RectMesh(Ne, Ne)
	space = H1Space(mesh, LagrangeBasis, p)

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

def convection(Ne, p):
	mesh = RectMesh(Ne, Ne)
	space = L2Space(mesh, LegendreBasis, p)
	Omega = np.array([1,0])

	C = Assemble(space, WeakConvectionIntegrator, Omega, (p-1)*p)
	M = Assemble(space, MassIntegrator, lambda x: .5, p*p) 
	F = FaceAssembleAll(space, UpwindTraceIntegrator, Omega, p*p)
	I = FaceAssembleRHS(space, InflowIntegrator, [Omega, lambda x,Omega: 1], p*p)

	A = F + C + M 

	psi = GridFunction(space)
	psi.data = spla.spsolve(A, I)

	return psi.L2Error(lambda x: np.exp(-.5*x[0]), 2*p+1)

def dgdiffusion(Ne, p):
	mesh = RectMesh(Ne, Ne)
	space = L2Space(mesh, LegendreBasis, p) 

	kappa = 100*10**p 
	K = Assemble(space, DiffusionIntegrator, lambda x: 1, 2*p+1)
	F = FaceAssembleAll(space, InteriorPenaltyIntegrator, kappa, 2*p+1)
	f = AssembleRHS(space, DomainIntegrator, lambda x: 2*np.pi**2*np.sin(np.pi*x[0])*np.sin(np.pi*x[1]), 2*p+1)

	A = K+F 
	T = GridFunction(space)
	T.data = spla.spsolve(A, f)

	err = T.L2Error(lambda x: np.sin(np.pi*x[0])*np.sin(np.pi*x[1]), 2*p+1)
	return err

def sn(Ne, p):
	mesh = RectMesh(Ne, Ne)
	space = L2Space(mesh, LegendreBasis, p)
	quad = LevelSym(4)
	sigma_t = lambda x: 1
	sigma_s = lambda x: .1 
	psi_ex, phi_ex, Q = TransportMMS(1, 1, 1, 10, 0, sigma_t, sigma_s)
	sweep = DirectSweeper(space, quad, sigma_t, sigma_s, Q, psi_ex, False)
	sn = Sn(sweep)
	psi = TVector(space, quad)
	phi = sn.SourceIteration(psi)

	return phi.L2Error(phi_ex, 2*p+1)

Ne = 5
@pytest.mark.parametrize('p', [1, 2, 3, 4])
@pytest.mark.parametrize('solver', [h1diffusion, convection, dgdiffusion, sn])
def test_ooa(solver, p):
	E1 = solver(Ne, p)
	E2 = solver(2*Ne, p)
	ooa = np.log2(E1/E2)
	if (abs(p+1-ooa)>.15):
		print('{:.3e}, {:.3e}'.format(E1, E2))
	assert(abs(p+1-ooa)<.15)
