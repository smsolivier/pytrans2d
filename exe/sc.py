#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from trans2d import * 
from scipy.integrate import solve_ivp 
from scipy.linalg import lu_factor, lu_solve 

from OutputCycler import OutputCycler
oc = OutputCycler()

def ForwardEuler(f, tspan, y0):
	Nx = 2**(p+1)
	dx = (tspan[1] - tspan[0])/Nx 
	x = tspan[0] 
	for n in range(Nx):
		y0 += dx * f(x, y0) 
		x += dx 

	return y0 

def SCSolve(el, trans, ip, Omega, b):
	global Nt 
	global Nc 
	Nc += 1
	def rhs(x,u):
		xi = np.array([x,u[0]])
		F = trans.F(xi)
		J = np.linalg.det(F) 
		X = trans.Transform(xi)
		q = el.CalcShape(xi)@b 
		return np.array([(Omega[1]*F[0,0] - Omega[0]*F[1,0])/(Omega[0]*F[1,1] - Omega[1]*F[0,1]), 
			J/(Omega[0]*F[1,1] - Omega[1]*F[0,1])*(q - u[1])])
	x0 = trans.Intersect(ip, -Omega)
	sol = solve_ivp(rhs, [x0[0], ip[0]], y0=[x0[1], 0], rtol=1e-13, atol=1e-13)
	Nt += len(sol.y[0,:])
	return sol.y[1,-1] 
	# return ForwardEuler(rhs, [x0[0], ip[0]], [x0[1], 0])[1]

def SCSolveSource(el, trans, ip, Omega, source):
	def rhs(x,u):
		xi = np.array([x,u[0]])
		F = trans.F(xi)
		J = np.linalg.det(F) 
		X = trans.Transform(xi)
		q = source(X) 
		return np.array([(Omega[1]*F[0,0] - Omega[0]*F[1,0])/(Omega[0]*F[1,1] - Omega[1]*F[0,1]), 
			J/(Omega[0]*F[1,1] - Omega[1]*F[0,1])*(q - u[1])])
	x0 = trans.Intersect(ip, -Omega)
	sol = solve_ivp(rhs, [x0[0], ip[0]], y0=[x0[1], inflow(x0, Omega)], rtol=1e-10, atol=1e-10)
	return sol.y[1,-1] 

def Exact(x):
	# return 16*x[0]*(1-x[0])*x[1]*(1-x[1]) + 1
	if (x[0]*Omega[1] < x[1]*Omega[0]):
		return np.exp(-1/Omega[0]*x[0])
		# return (inflow(x,Omega) + Omega[0])*np.exp(-1/Omega[0]*x[0]) + x[0] - Omega[0]
	else:
		return np.exp(-1/Omega[1]*x[1])
		# return (inflow(x,Omega) - x[0] + Omega[0]/Omega[1]*x[1] + Omega[0])*np.exp(-x[1]/Omega[1]) + x[0] - Omega[0]

p = oc.GetOpt(0,3) 
N = 1 
h = 1
mesh = RectMesh(N,N, [0,0], [h,h])
trans = mesh.trans[0]
orders = np.arange(1, 12)
Nc = 0 
Nt = 0
it1 = [] 
it2 = []
it3 = [] 
Nt1 = [] 
Nt2 = [] 
for p in orders:
# for p in [p]:
	space = L2Space(mesh, LegendreBasis, p) 
	m = MomentBasis(space.el.basis) 

	Omega = np.array([1,.25])
	inflow = lambda x, Omega: 0
	G = Assemble(space, WeakConvectionIntegrator, Omega, 2*p+1)
	M = Assemble(space, MassIntegrator, lambda x: 1, 2*p+1) 
	F = FaceAssembleAll(space, UpwindTraceIntegrator, Omega, 2*p+1)
	I = FaceAssembleRHS(space, InflowIntegrator, [Omega, inflow], 2*p+1)
	# source = lambda x: 16*(Omega[0]*(1-2*x[0])*(x[1]-x[1]**2) + 
		# Omega[1]*(x[0]-x[0]**2)*(1-2*x[1]) + (x[0]-x[0]**2)*(x[1] - x[1]**2))+1
	source = lambda x: x[0]
	q = AssembleRHS(space, DomainIntegrator, source, 2*p+1)
	I += q 

	A = G + M + F
	it = 0
	norm = 0 
	def Callback(r):
		global it 
		global norm 
		global prev 
		it += 1 
		norm = np.linalg.norm(r)
		# print('it = {:3}, norm = {:.3e}'.format(it, norm))

	u = GridFunction(space)
	M2 = MassIntegrator(space.el, trans, lambda x: 1, 2*p+1)
	el_lo = Element(LegendreBasis, p)
	Mlo = MassIntegrator(el_lo, trans, lambda x: 1, 2*p+1)
	mM = MixMassIntegrator(el_lo, space.el, trans, lambda x: 1, 2*p+1)
	luM2 = lu_factor(M2)
	plot = False
	def SCProj(b):
		Mb = lu_solve(luM2, b)
		r = np.zeros(M2.shape[0])
		# ip, w = quadrature.GetLumped(space.el) 
		ip, w = quadrature.Get(2*p+2)
		for n in range(len(w)):
			r += space.el.CalcShape(ip[n]) * SCSolve(space.el, mesh.trans[0], ip[n], Omega, Mb) * w[n] * trans.Jacobian(ip[n]) 

		sol = r/M2.diagonal() if type(space.el.basis)==LegendreBasis else lu_solve(luM2, r) 

		if (plot):
			err = 0 
			ip, w = quadrature.Get(2*p+2) 
			for n in range(len(w)):
				err += (space.el.Interpolate(trans, ip[n], sol)
					- SCSolve(space.el, mesh.trans[0], ip[n], Omega, Mb))**2 * w[n] * trans.Jacobian(ip[n])

			plt.figure(figsize=(12,4))
			plt.subplot(1,2,1)
			X,Y,U = space.el.Plot(trans, Mb)
			plt.pcolor(X,Y, U)
			plt.colorbar()
			plt.title('norm = {:.3e}'.format(np.linalg.norm(Mb)))

			plt.subplot(1,2,2)
			X,Y,U = space.el.Plot(trans, sol)
			plt.pcolor(X,Y, U)
			plt.colorbar()
			plt.title('err = {:.3e}'.format(np.sqrt(err)))

		return sol

	def SCEval(b):
		Mb = lu_solve(luM2, b)
		r = np.zeros(space.el.Nn)
		for n in range(space.el.Nn):
			r[n] = SCSolve(space.el, mesh.trans[0], space.el.nodes[n], Omega, Mb)

		if (plot):
			err = 0 
			ip, w = quadrature.Get(2*p+2)
			# ip, w = quadrature.GetLumped(space.el)
			for n in range(len(w)):
				err += (space.el.Interpolate(trans, ip[n], r)
					- SCSolve(space.el, mesh.trans[0], ip[n], Omega, Mb))**2 * w[n] * trans.Jacobian(ip[n])

			plt.figure(figsize=(12,4))
			plt.subplot(1,2,1)
			X,Y,U = space.el.Plot(trans, Mb)
			plt.pcolor(X,Y, U)
			plt.colorbar()
			plt.title('norm = {:.3e}'.format(np.linalg.norm(Mb)))

			plt.subplot(1,2,2)
			X,Y,U = space.el.Plot(trans, r)
			plt.pcolor(X,Y, U)
			plt.colorbar()
			plt.title('err = {:.3e}'.format(np.sqrt(err)))

		return r 

	M = spla.LinearOperator(A.shape, SCEval)
	plot = False
	Nt = 0
	Nc = 0 
	res = []
	u.data, info = pyamg.krylov.gmres(A.tocsc(), I, M=M,
		callback=Callback, tol=1e-10, maxiter=A.shape[0], residuals=res)
	Nt /= Nc
	print('gmres: it = {}/{}, residual = {:.3e}, Nt = {:.2f}, Nc = {}'.format(it, space.el.Nn, 
		np.linalg.norm(A*u.data - I), Nt, Nc))
	it1.append(it) 
	Nt1.append(Nt)
	it = 0 
	Nt = 0 
	Nc = 0
	u.data, info = pyamg.krylov.bicgstab(A.tocsc(), I, M=M, 
		callback=Callback, tol=1e-10, maxiter=A.shape[0])
	Nt /= Nc
	print('bicg: it = {}/{}, residual = {:.3e}, Nt = {:.2f}, Nc = {}'.format(it, space.el.Nn, 
		np.linalg.norm(A*u.data - I), Nt, Nc))
	it2.append(it)
	Nt2.append(Nt)
	# u.data = np.zeros(space.el.Nn)
	# Nt = 0
	# for it in range(100):
	# 	res = I - A*u.data
	# 	norm = np.linalg.norm(res) 
	# 	# print(it,norm)
	# 	if (norm<1e-10):
	# 		break 
	# 	u.data += SCEval(res) 
	# print('p = {}, it = {}/{}, residual = {:.3e}, Nt = {}'.format(p, it, space.el.Nn, 
	# 	np.linalg.norm(A*u.data - I), Nt))
	# it3.append(it)

	# print('err = {:.3e}'.format(u.L2Error(Exact, 2*p+2)))

plt.figure()
plt.plot(orders, it1, '-o', label='GMRES')
plt.plot(orders, it2, '-o', label='BiCGStab')
# plt.plot(orders, it3, '-o', label='SLI')
plt.xlabel('$p$')
plt.ylabel('Iterations')
plt.legend()

plt.figure()
plt.semilogy(orders, Nt1, '-o', label='GMRES')
plt.semilogy(orders, Nt2, '-o', label='BiCGStab')
# plt.plot(orders, it3, '-o', label='SLI')
plt.xlabel('$p$')
plt.ylabel('Avg. Number of RK45 Time Steps')
plt.legend()
plt.show()

# err = 0
# for n in range(len(w)):
# 	x = trans.Transform(ip[n])
# 	err += (Exact(x) - SCSolve(space.el, mesh.trans[0], ip[n], Omega, np.zeros(space.el.Nn)))**2 * w[n] 

# print('err = {:.3e}'.format(np.sqrt(err)))

# xi_1d = np.linspace(-1,1, 80)
# xi, eta = np.meshgrid(xi_1d, xi_1d) 
# x = np.zeros(xi.shape)
# y = np.zeros(eta.shape) 
# U = np.zeros((len(xi), len(xi)))
# S = np.zeros((len(xi), len(xi)))
# for i in range(len(xi)):
# 	for j in range(len(xi)):
# 		x[i,j], y[i,j] = mesh.trans[0].Transform([xi[i,j], eta[i,j]])
# 		U[i,j] = u.Interpolate(0, [xi[i,j], eta[i,j]]) 
# 		# U[i,j] = Exact([x[i,j], y[i,j]])
# 		# U[i,j] = np.fabs(Exact([x[i,j], y[i,j]]) 
# 			# - u.Interpolate(0, [xi[i,j], eta[i,j]]))
# 		# U[i,j] = SCSolve(space.el, mesh.trans[0], [xi[i,j], eta[i,j]], Omega, np.zeros(space.el.Nn))

# plt.pcolor(x,y, U)
# plt.colorbar()
# plt.show()
