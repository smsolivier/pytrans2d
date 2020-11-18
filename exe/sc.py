#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from trans2d import * 
from scipy.integrate import solve_ivp 

from OutputCycler import OutputCycler
oc = OutputCycler()

def RayTrace(trans, ip, Omega):
	found = False 
	IP = trans.Transform(ip) 
	for i in range(2):
		xi = np.zeros(2)
		xi[i] = -1 
		for n in range(25):
			F = trans.F(xi)
			x = trans.Transform(xi) 
			rhs = IP - x + np.dot(F, xi)
			rhs = np.append(rhs, -1) 
			A = np.zeros((3,3))
			A[:2,:2] = F 
			A[:2,-1] = Omega 
			A[-1,i] = 1 

			u = np.linalg.solve(A, rhs) 
			xi = u[:2] 
			t = u[-1] 
			norm = np.linalg.norm(trans.Transform(xi) - (IP - t*Omega))
			if (norm<1e-8):
				break 

		if (xi[(i+1)%2] >= -1. and xi[(i+1)%2] <= 1. and t>=0 and norm<1e-7):
			found = True
			break 

	if not(found):
		raise RuntimeError('intersection not found. ip = ({:.2f},{:.2f}), J = {:.3f}'.format(
			ip[0], ip[1], trans.Jacobian(ip))) 

	return xi 

def ForwardEuler(f, tspan, y0):
	Nx = 2**(p+1)
	dx = (tspan[1] - tspan[0])/Nx 
	x = tspan[0] 
	for n in range(Nx):
		y0 += dx * f(x, y0) 
		x += dx 

	return y0 

def SCSolve(el, trans, ip, Omega, b):
	def rhs(x,u):
		xi = np.array([x,u[0]])
		F = trans.F(xi)
		J = np.linalg.det(F) 
		X = trans.Transform(xi)
		q = PolyVal2D(m.B, m.B, xi)@b
		# q = el.CalcShape(xi)@b 
		return np.array([(Omega[1]*F[0,0] - Omega[0]*F[1,0])/(Omega[0]*F[1,1] - Omega[1]*F[0,1]), 
			J/(Omega[0]*F[1,1] - Omega[1]*F[0,1])*(q - u[1])])
	x0 = RayTrace(trans, ip, Omega)
	# sol = solve_ivp(rhs, [x0[0], ip[0]], y0=[x0[1], 0], rtol=1e-12, atol=1e-12)
	# return sol.y[1,-1] 
	return ForwardEuler(rhs, [x0[0], ip[0]], [x0[1], 0])[1]

def SCSolveSource(el, trans, ip, Omega, source):
	def rhs(x,u):
		xi = np.array([x,u[0]])
		F = trans.F(xi)
		J = np.linalg.det(F) 
		X = trans.Transform(xi)
		q = source(X) 
		return np.array([(Omega[1]*F[0,0] - Omega[0]*F[1,0])/(Omega[0]*F[1,1] - Omega[1]*F[0,1]), 
			J/(Omega[0]*F[1,1] - Omega[1]*F[0,1])*(q - u[1])])
	x0 = RayTrace(trans, ip, Omega)
	sol = solve_ivp(rhs, [x0[0], ip[0]], y0=[x0[1], inflow(x0, Omega)], rtol=1e-10, atol=1e-10)
	return sol.y[1,-1] 

def Exact(x):
	# return 16*x[0]*(1-x[0])*x[1]*(1-x[1])
	if (x[0]*Omega[1] < x[1]*Omega[0]):
		# return np.exp(-1/Omega[0]*x[0])
		return (inflow(x,Omega) + Omega[0])*np.exp(-1/Omega[0]*x[0]) + x[0] - Omega[0]
	else:
		# return np.exp(-1/Omega[1]*x[1])
		return (inflow(x,Omega) - x[0] + Omega[0]/Omega[1]*x[1] + Omega[0])*np.exp(-x[1]/Omega[1]) + x[0] - Omega[0]

p = oc.GetOpt(0,3) 
N = 1 
mesh = RectMesh(N,N, [0,0], [1,1])
trans = mesh.trans[0]
orders = np.arange(1, 12)
itse = [] 
itsp = []
for p in orders:
	space = L2Space(mesh, LegendreBasis, p) 
	m = MomentBasis(space.el.basis) 

	Omega = np.array([1,.25])
	inflow = lambda x, Omega: 0
	G = Assemble(space, WeakConvectionIntegrator, Omega, 2*p+1)
	M = Assemble(space, MassIntegrator, lambda x: 1, 2*p+1) 
	F = FaceAssembleAll(space, UpwindTraceIntegrator, Omega, 2*p+1)
	I = FaceAssembleRHS(space, InflowIntegrator, [Omega, inflow], 2*p+1)
	# source = lambda x: 16*(Omega[0]*(1-2*x[0])*(x[1]-x[1]**2) + Omega[1]*(x[0]-x[0]**2)*(1-2*x[1]) + (x[0]-x[0]**2)*(x[1] - x[1]**2))
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
	ip, w = quadrature.Get(2*p+1)
	M2 = MassIntegrator(space.el, mesh.trans[0], lambda x: 1, 2*p+1)
	def SCProj(b):
		r = np.zeros(M2.shape[0])
		for n in range(len(w)):
			r += space.el.CalcShape(ip[n]) * SCSolve(space.el, mesh.trans[0], ip[n], Omega, b) * w[n] * trans.Jacobian(ip[n]) 

		return r/M2.diagonal() if type(space.el.basis)==LegendreBasis else np.linalg.solve(M2, r) 

	def SCEval(b):
		r = np.zeros(space.el.Nn)
		for n in range(space.el.Nn):
			r[n] = SCSolve(space.el, mesh.trans[0], space.el.nodes[n], Omega, b)

		# num = 50
		# xi = np.linspace(-1,1,num)
		# X = np.zeros((num,num))
		# Y = np.zeros((num,num))
		# U = np.zeros((num,num))
		# for i in range(num):
		# 	for j in range(num):
		# 		ip = np.array([xi[i], xi[j]])
		# 		X[i,j], Y[i,j] = trans.Transform(ip)
		# 		# U[i,j] = space.el.CalcShape(ip)@b
		# 		U[i,j] = SCSolve(space.el, mesh.trans[0], ip, Omega, b)

		# plt.pcolor(X,Y,U)
		# plt.colorbar()
		# plt.show()

		return r 

	# u.data = SCProj(np.zeros(space.el.Nn))
	# u.data = SCEval(np.zeros(space.el.Nn))
	# u.data, info = pyamg.krylov.fgmres(A.tocsc(), I,
	# 	callback=Callback, tol=1e-10, maxiter=A.shape[0])
	# print('it = {}/{}, residual = {:.3e}'.format(it, space.el.Nn, np.linalg.norm(A*u.data - I)))
	# itse.append(it) 
	# it = 0 
	# u.data, info = pyamg.krylov.fgmres(A.tocsc(), I, M=spla.LinearOperator(A.shape, SCEval), 
		# callback=Callback, tol=1e-10, maxiter=A.shape[0])
	u.data, info = pyamg.krylov.steepest_descent(A.tocsc(), I, M=spla.LinearOperator(A.shape, SCEval), 
		callback=Callback, tol=1e-10, maxiter=A.shape[0])
	print('p = {}, it = {}/{}, residual = {:.3e}'.format(p, it, space.el.Nn, np.linalg.norm(A*u.data - I)))
	itsp.append(it)


	# u.data = np.linalg.solve(A.todense(), I)

	# print('proj. err = {:.3e}'.format(u.L2Error(Exact, 2*p+1)))

# plt.plot(orders, itse, '-o', label='Unprec')
plt.plot(orders, itsp, '-o', label='SC')
# plt.plot(orders, (orders+1)**2, '-o', label='Size of System')
plt.xlabel('$p$')
plt.ylabel('FGMRES Iterations')
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
