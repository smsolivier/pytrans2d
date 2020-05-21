#!/usr/bin/env python3

import numpy as np
import warnings 
import pytest 

from trans2d import * 

def TransportMMS(alpha, beta, gamma, delta, eta, sigt, sigs):
	f = lambda x: np.sin(np.pi*x[0])*np.sin(np.pi*x[1]) + delta 
	df = lambda x: np.array([np.pi*np.cos(np.pi*x[0])*np.sin(np.pi*x[1]), 
		np.pi*np.sin(np.pi*x[0])*np.cos(np.pi*x[1])])
	h = lambda x: np.sin(2*np.pi*x[0])*np.sin(2*np.pi*x[1])
	dh = lambda x: np.array([2*np.pi*np.cos(2*np.pi*x[0])*np.sin(2*np.pi*x[1]), 
		2*np.pi*np.sin(2*np.pi*x[0])*np.cos(2*np.pi*x[1])])
	# h = lambda x: x[0]**3-3/2*x[0]**2+.5*x[0]
	# dh = lambda x: 3*x[0]**2 - 3*x[0] + .5 
	# g = lambda x: x[0]*(1-x[0])*x[1]*(1-x[1])
	# dg = lambda x: np.array([(1-2*x[0])*x[1]*(1-x[1]), x[0]*(1-x[0])*(1-2*x[1])])
	L = 1 + 2*eta
	g = lambda x: np.sin(np.pi*(x[0]+eta)/L)*np.sin(np.pi*(x[1]+eta)/L)
	dg = lambda x: np.array([np.pi/L*np.cos(np.pi*(x[0]+eta)/L)*np.sin(np.pi*(x[1]+eta)/L), 
		np.pi/L*np.sin(np.pi*(x[0]+eta)/L)*np.cos(np.pi*(x[1]+eta)/L)])
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

	K = Assemble(space, DiffusionIntegrator, lambda x: 1, 2*p)
	M = Assemble(space, MassIntegrator, lambda x: .1, 2*p+1)
	f = AssembleRHS(space, DomainIntegrator, lambda x: (2*np.pi**2 + .1)*np.sin(np.pi*x[0])*np.sin(np.pi*x[1]), 2*p+1)
	A = K+M

	A = A.tolil()
	A[space.bnodes,:] = 0 
	A[space.bnodes,space.bnodes] = 1
	f[space.bnodes] = 0 

	T = GridFunction(space)
	T.data = spla.spsolve(A.tocsc(), f)

	err = T.L2Error(lambda x: np.sin(np.pi*x[0])*np.sin(np.pi*x[1]), 2*p+2)
	return err 

def mixdiffusion(Ne, p):
	mesh = RectMesh(Ne, Ne)
	phi_space = L2Space(mesh, LegendreBasis, p)
	J_space = H1Space(mesh, LobattoBasis, p+1, 2)
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
	return phi.L2Error(phi_ex, 2*p+2)

def rtdiffusion(Ne, p):
	mesh = RectMesh(Ne, Ne)
	rt = RTSpace(mesh, LobattoBasis, LegendreBasis, p)
	l2 = L2Space(mesh, LegendreBasis, p) 
	M = Assemble(rt, VectorFEMassIntegrator, lambda x: -1, 2*(p+2)+1)
	D = MixAssemble(l2, rt, VectorFEDivIntegrator, 1, 2*(p+1)+1)
	Q = lambda x: 2*np.pi**2*np.sin(np.pi*x[0])*np.sin(np.pi*x[1])
	f = AssembleRHS(l2, DomainIntegrator, Q, 2*(p+1)+1)

	A = sp.bmat([[M, D.transpose()], [D, None]]).tocsc()
	rhs = np.concatenate((np.zeros(rt.Nu), f))
	x = spla.spsolve(A, rhs) 
	T = GridFunction(l2)
	T.data = x[rt.Nu:]
	return T.L2Error(lambda x: np.sin(np.pi*x[0])*np.sin(np.pi*x[1]), 2*(p+1)+2)

def convection(Ne, p):
	mesh = RectMesh(Ne, Ne)
	space = L2Space(mesh, LegendreBasis, p)
	Omega = np.array([1,0])

	C = Assemble(space, WeakConvectionIntegrator, Omega, 2*p)
	M = Assemble(space, MassIntegrator, lambda x: .5, 2*p+1) 
	F = FaceAssembleAll(space, UpwindTraceIntegrator, Omega, 2*p+1)
	I = FaceAssembleRHS(space, InflowIntegrator, [Omega, lambda x,Omega: 1], 2*p+1)

	A = F + C + M 

	psi = GridFunction(space)
	psi.data = spla.spsolve(A, I)

	return psi.L2Error(lambda x: np.exp(-.5*x[0]), 2*p+2)

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

	err = T.L2Error(lambda x: np.sin(np.pi*x[0])*np.sin(np.pi*x[1]), 2*p+2)
	return err

def sn_direct(Ne, p):
	mesh = RectMesh(Ne, Ne)
	space = L2Space(mesh, LegendreBasis, p)
	quad = LevelSym(4)
	sigma_t = lambda x: 1
	sigma_s = lambda x: .1 
	psi_ex, phi_ex, Q = TransportMMS(1, 1, 1, 10, .1, sigma_t, sigma_s)
	sweep = DirectSweeper(space, quad, sigma_t, sigma_s, Q, psi_ex, False)
	sn = Sn(sweep)
	psi = TVector(space, quad)
	phi = sn.SourceIteration(psi)
	res = sweep.ComputeResidual(psi, phi) 
	if (res>1e-10):
		warnings.warn('sweep residual = {:.3e}'.format(res), stacklevel=2)

	return phi.L2Error(phi_ex, 2*p+2)

def sn_sweep(Ne, p):
	mesh = RectMesh(Ne, Ne)
	space = L2Space(mesh, LegendreBasis, p)
	quad = LevelSym(4)
	sigma_t = lambda x: 1
	sigma_s = lambda x: .1 
	psi_ex, phi_ex, Q = TransportMMS(1, 1, 1, 10, .1, sigma_t, sigma_s)
	sweep = Sweeper(space, quad, sigma_t, sigma_s, Q, psi_ex, False)
	sn = Sn(sweep)
	psi = TVector(space, quad)
	phi = sn.SourceIteration(psi)
	res = sweep.ComputeResidual(psi, phi) 
	if (res>1e-10):
		warnings.warn('sweep residual = {:.3e}'.format(res), stacklevel=2)

	return phi.L2Error(phi_ex, 2*p+2)

def p1(Ne, p):
	mesh = RectMesh(Ne, Ne)
	space = L2Space(mesh, LegendreBasis, p)
	quad = LevelSym(4)
	sigma_t = lambda x: 1
	sigma_s = lambda x: .1
	psi_ex, phi_ex, Q = TransportMMS(1, .1, 0, 0, 0, sigma_t, sigma_s)
	sweep = AbstractSweeper(space, quad, sigma_t, sigma_s, Q, psi_ex, True)
	sn = P1SA(sweep)
	f = np.zeros(sn.space.Nu)
	g = np.zeros(sn.J_space.Nu)
	for a in range(quad.N):
		Omega = quad.Omega[a]
		w = quad.w[a] 
		f += AssembleRHS(space, DomainIntegrator, lambda x: Q(x,Omega)*w, 2*p+2)
		g += AssembleRHS(sn.J_space, VectorDomainIntegrator, lambda x: Omega*Q(x,Omega)*w, 2*p+2)

	rhs = np.concatenate((3*g,f))
	x = sn.lu.solve(rhs)
	phi = GridFunction(space)
	phi.data = x[sn.J_space.Nu:]
	return phi.L2Error(phi_ex, 2*p+2)

def p1sa(Ne, p):
	mesh = RectMesh(Ne, Ne)
	space = L2Space(mesh, LobattoBasis, p)
	quad = LevelSym(4)
	sigma_t = lambda x: 1
	sigma_s = lambda x: .1
	psi_ex, phi_ex, Q = TransportMMS(1, .1, 1, 1, .1, sigma_t, sigma_s)
	sweep = Sweeper(space, quad, sigma_t, sigma_s, Q, psi_ex, False)
	sn = P1SA(sweep)
	psi = TVector(space, quad)
	phi = sn.SourceIteration(psi)
	res = sweep.ComputeResidual(psi, phi) 
	if (res>1e-10):
		warnings.warn('sweep residual = {:.3e}'.format(res), stacklevel=2)

	return phi.L2Error(phi_ex, 2*p+2)

def vef(Ne, p):
	mesh = RectMesh(Ne, Ne)
	space = L2Space(mesh, LegendreBasis, p)
	phi_space = L2Space(mesh, LegendreBasis, p)
	Jspace = H1Space(mesh, LobattoBasis, p+1, 2)
	quad = LevelSym(4)
	eps = 1e-1
	sigma_t = lambda x: 1/eps
	sigma_s = lambda x: 1/eps - eps 
	psi_ex, phi_ex, Q = TransportMMS(1, 1, 1, 10, .1, sigma_t, sigma_s)
	sweep = AbstractSweeper(space, quad, sigma_t, sigma_s, Q, psi_ex, False)
	sn = VEF(phi_space, Jspace, sweep)
	psi = TVector(space, quad)
	psi.Project(psi_ex)
	with warnings.catch_warnings():
		warnings.filterwarnings('ignore', category=NegativityWarning)
		phi, J = sn.Mult(psi)

	return phi.L2Error(phi_ex, 2*p+2)	

Ne = 3
@pytest.mark.parametrize('p', [1, 2, 3, 4])
@pytest.mark.parametrize('solver', [h1diffusion, mixdiffusion, rtdiffusion, convection, dgdiffusion, 
	sn_direct, sn_sweep, p1, p1sa, vef])
def test_ooa(solver, p):
	E1 = solver(Ne, p)
	E2 = solver(2*Ne, p)
	ooa = np.log2(E1/E2)
	print('p={:.3f} ({:.3e}, {:.3e})'.format(ooa, E1, E2))
	assert(abs(p+1-ooa)<.15)
