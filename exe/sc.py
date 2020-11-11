#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from trans2d import * 
from scipy.integrate import solve_ivp 

def RayTrace(trans, ip, Omega):
	found = False 
	for i in range(2):
		xi = np.zeros(2)
		xi[i] = -1 
		# def RefTracing(x,y):
		# 	F = trans.F([x,y[0]])
		# 	q = (Omega[1]*F[0,0] - Omega[0]*F[1,0])/(Omega[0]*F[1,1] - Omega[1]*F[0,1])
		# 	return q 
		# span = np.ones(2)*-1 
		# span[i] = ip[i]
		# sol = solve_ivp(RefTracing, span, y0=[ip[(i+1)%2]], atol=1e-3)
		# if (i==0):
		# 	xi[0] = -1
		# 	xi[1] = sol.y[0,-1]
		# else:
		# 	xi[1] = -1 
		# 	xi[0] = sol.y[0,-1] 
		for n in range(25):
			F = trans.F(xi)
			x = trans.Transform(xi) 
			rhs = ip - x + np.dot(F, xi)
			rhs = np.append(rhs, -1) 
			A = np.zeros((3,3))
			A[:2,:2] = F 
			A[:2,-1] = Omega 
			A[-1,i] = 1 

			u = np.linalg.solve(A, rhs) 
			xi = u[:2] 
			t = u[-1] 
			norm = np.linalg.norm(trans.Transform(xi) - (ip - t*Omega))
			if (norm<1e-8):
				break 

		if (xi[(i+1)%2] >= -1. and xi[(i+1)%2] <= 1. and t>=0 and norm<1e-8):
			found = True
			break 

	if not(found):
		raise RuntimeError('intersection not found') 

	return xi 

def ForwardEuler(f, tspan, y0):
	Nx = 100
	dx = (tspan[1] - tspan[0])/Nx 
	x = tspan[0] 
	for n in range(Nx):
		y0 += dx * f(x, y0) 
		x += dx 

	return y0 

def SCSolve(el, trans, ip, Omega, b):
	def rhs(x,u):
		F = trans.F([x,u[0]])
		J = np.linalg.det(F) 
		return np.array([(Omega[1]*F[0,0] - Omega[0]*F[1,0])/(Omega[0]*F[1,1] - Omega[1]*F[0,1]), 
			J/(F[0,1]*Omega[1] - F[1,1]*Omega[0])*u[1] + el.CalcShape([x,u[0]])@b])
	x0 = RayTrace(trans, ip, Omega)
	sol = solve_ivp(rhs, [x0[0], ip[0]], y0=[x0[1], 1], rtol=1e-10)
	return sol.y[1,-1] 
	# return ForwardEuler(rhs, [x0[0], ip[0]], [x0[1], 1])[1]

def Exact(x):
	if (x[0]*Omega[0] < x[1]*Omega[1]):
		return np.exp(-1/Omega[0]*(x[0]+1))
	else:
		return np.exp(-1/Omega[1]*(x[1]+1))

p = 3
N = 1 

mesh = RectMesh(N,N, [-1,-1], [1,1])
space = L2Space(mesh, LegendreBasis, p) 

Omega = np.array([1,1])
inflow = 1
G = Assemble(space, WeakConvectionIntegrator, Omega, 2*p+1)
M = Assemble(space, MassIntegrator, lambda x: 1, 2*p+1) 
F = FaceAssembleAll(space, UpwindTraceIntegrator, Omega, 2*p+1)
I = FaceAssembleRHS(space, InflowIntegrator, [Omega, lambda x, Omega: inflow], 2*p+1)

A = G + M + F
it = 0
norm = 0 
def Callback(r):
	global it 
	global norm 
	it += 1 
	norm = np.linalg.norm(r) 
	print('it = {:3}, norm = {:.3e}'.format(it, norm))

u = GridFunction(space)
ip, w = quadrature.Get(3*p+1)
M = MassIntegrator(space.el, mesh.trans[0], lambda x: 1, 2*p+1)
def SCProj(b):
	r = np.zeros(M.shape[0])
	for n in range(len(w)):
		r += space.el.CalcShape(ip[n]) * SCSolve(space.el, mesh.trans[0], ip[n], Omega, b) * w[n] 

	return np.linalg.solve(M, r) 

def SCEval(b):
	r = np.zeros(space.el.Nn)
	for n in range(space.el.Nn):
		r[n] = SCSolve(space.el, mesh.trans[0], space.el.nodes[n], Omega, b)

	return r 

# u.data = SCProj(np.zeros(space.el.Nn))
# u.data = SCEval(np.zeros(space.el.Nn))
# u.data, info = pyamg.krylov.gmres(A.tocsc(), I, M=spla.LinearOperator(A.shape, SCProj), 
	# callback=Callback, tol=1e-10, maxiter=A.shape[0])
# print(np.linalg.norm(A*u.data - I))
u.data = np.linalg.solve(A.todense(), I)

print('proj. err = {:.3e}'.format(u.L2Error(Exact, p+1)))
err = 0
for n in range(len(w)):
	err += (Exact(ip[n]) - SCSolve(space.el, mesh.trans[0], ip[n], Omega, np.zeros(space.el.Nn)))**2 * w[n] 

print('err = {:.3e}'.format(err))

xi_1d = np.linspace(-1,1, 20)
xi, eta = np.meshgrid(xi_1d, xi_1d) 
U = np.zeros((len(xi), len(xi)))
S = np.zeros((len(xi), len(xi)))
for i in range(len(xi)):
	for j in range(len(xi)):
		# U[i,j] = u.Interpolate(0, [xi[i,j], eta[i,j]]) 
		# U[i,j] = Exact([xi[i,j], eta[i,j]])
		U[i,j] = np.fabs(Exact([xi[i,j], eta[i,j]]) 
			- u.Interpolate(0, [xi[i,j], eta[i,j]]))
		# U[i,j] = SCSolve(space.el, mesh.trans[0], [xi[i,j], eta[i,j]], Omega, np.zeros(space.el.Nn))

		# rhs = lambda x,u: [Omega[1]/Omega[0], -1./Omega[0]*u[1]]
		# x0 = RayTrace(mesh.trans[0], np.array([xi[i,j], eta[i,j]]), Omega)
		# sol = solve_ivp(rhs, [x0[0], xi[i,j]], y0=[x0[1], 1])
		# S[i,j] = sol.y[1,-1]

plt.pcolor(xi,eta, U)
plt.colorbar()
plt.show()
