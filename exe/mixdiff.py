#!/usr/bin/env python3

import numpy as np
import sys 
import warnings
import matplotlib.pyplot as plt

from trans2d import * 

warnings.simplefilter('ignore', ToleranceWarning)

Ne = 10 
p = 3
if (len(sys.argv)>1):
	Ne = int(sys.argv[1])
if (len(sys.argv)>2):
	p = int(sys.argv[2])

mesh = RectMesh(Ne, Ne)
J_space = H1Space(mesh, LagrangeBasis, p, 2)
phi_space = L2Space(mesh, LegendreBasis, p-1, 1, False)

print('Nu = {}'.format(J_space.Nu + phi_space.Nu))

sigma_t = lambda x: 1000*(x[0]>.5) + 1
sigma_a = lambda x: 1*(x[0]<.5) 
# sigma_t = lambda x: 1 
# sigma_a = lambda x: .1

qorder = 2*p+2
Mt = -3*Assemble(J_space, VectorMassIntegrator, sigma_t, qorder)
Mtl = -3*Assemble(J_space, VectorMassIntegratorRowSum, sigma_t, qorder)
Ma = Assemble(phi_space, MassIntegrator, sigma_a, qorder)
D = MixAssemble(phi_space, J_space, MixDivIntegrator, 1, qorder)
f = AssembleRHS(phi_space, DomainIntegrator, lambda x: 1, qorder)

Mtinv = sp.diags(1/Mtl.diagonal())
S = Ma - D*Mtinv*D.transpose()

Jlow = J_space.LORefine()
phi_low = L2Space(Jlow.mesh, LegendreBasis, 0, 1, False) 
Mt_low = -3*Assemble(Jlow, VectorMassIntegrator, sigma_t, qorder) 
Mtl_low = -3*Assemble(Jlow, VectorMassIntegratorRowSum, sigma_t, qorder)
D_low = MixAssemble(phi_low, Jlow, MixDivIntegrator, 1, qorder)
Ma_low = Assemble(phi_low, MassIntegrator, sigma_a, qorder) 
Mtinv_low = sp.diags(1/Mtl_low.diagonal()) 
Slow = Ma_low - D_low*Mtinv_low*D_low.transpose()

M = sp.bmat([[Mt, D.transpose()], [D, Ma]])
rhs = np.concatenate((np.zeros(J_space.Nu), f))

Slow_inv = spla.inv(Slow)
prec = Slow_inv * S 
kappa = spla.norm(spla.inv(prec))*spla.norm(prec)
print('kappa(prec) = {:.3e}'.format(kappa))
kappa = np.linalg.cond(prec.todense())
print('kappa(M) = {:.3e}'.format(kappa))

# amg1 = pyamg.ruge_stuben_solver(S.tocsr())
# # amg2 = pyamg.ruge_stuben_solver(Slow.tocsr())

# # r1 = []
# # amg1.solve(f, tol=1e-10, maxiter=10000, residuals=r1)
# # print('amg: it={}, norm={:.3e}'.format(len(r1), r1[-1]))
# # r2 = []
# # amg2.solve(f, tol=1e-10, maxiter=10000, residuals=r2)
# # print('low: it={}, norm={:.3e}'.format(len(r2), r2[-1]))

# inner = 1
# solver = AMGSolver(1e-10, 1000, inner, False)
# phi = GridFunction(phi_space)
# phi.data = solver.Solve(S, S, f)
# print('amg: it={:4}, norm={:.3e}'.format(solver.it, solver.norm))

# solver = AMGSolver(1e-10, 1000, inner, False)
# solver.Solve(Slow, Slow, f)
# print('amg: it={:4}, norm={:.3e}'.format(solver.it, solver.norm))

# mesh.WriteVTK('solution', cell={'phi':phi.ElementData()})