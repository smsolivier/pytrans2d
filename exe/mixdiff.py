#!/usr/bin/env python3

import numpy as np
import sys 
import warnings

from trans2d import * 

warnings.simplefilter('ignore', ToleranceWarning)

Ne = 10 
p = 3
if (len(sys.argv)>1):
	Ne = int(sys.argv[1])
if (len(sys.argv)>2):
	p = int(sys.argv[2])

mesh = RectMesh(Ne, Ne)
J_space = H1Space(mesh, LobattoBasis, p, 2)
phi_space = L2Space(mesh, LegendreBasis, p-1, 1, False)

print('Nu = {}'.format(J_space.Nu + phi_space.Nu))

sigma_t = lambda x: 1
sigma_a = lambda x: .1

qorder = 2*p+1
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
Ma_low = Assemble(phi_low, MassIntegrator, sigma_a, qorder)
Dlow = MixAssemble(phi_low, Jlow, MixDivIntegrator, 1, qorder)

Mtinv_low = sp.diags(1/Mtl_low.diagonal())
Slow = Ma_low - Dlow*Mtinv_low*Dlow.transpose()

M = sp.bmat([[Mt, D.transpose()], [D, Ma]])
rhs = np.concatenate((np.zeros(J_space.Nu), f))

solver = AMGSolver(1e-10, 1000, 1, False)
solver.Solve(S, S, f)
print('amg: it={:4}, norm={:.3e}'.format(solver.it, solver.norm))

solver.Solve(S, Slow, f)
print('low: it={:4}, norm={:.3e}'.format(solver.it, solver.norm))
# solver = BlockLDU(1e-10, 1000, 1, False)
# x = solver.Solve(Mt, D.transpose(), D, Ma, Mtinv, S, M, rhs)
# print('gmres: it={}, norm={:.3e}'.format(solver.it, solver.norm))

# x = solver.Solve(Mt, D.transpose(), D, Ma, Mtinv, Slow, M, rhs)
# print('low S: it={}, norm={:.3e}'.format(solver.it, solver.norm))

# x = solver.Solve(Mt, D.transpose(), D, Ma, Mtinv, Slow2, M, rhs)
# print('low S2: it={}, norm={:.3e}'.format(solver.it, solver.norm))

# x = solver.Solve(Mt, D.transpose(), D, Ma, Mtinv_low, Slow, M, rhs)
# print('low D: it={}, norm={:.3e}'.format(solver.it, solver.norm))

# x = solver.Solve(Mt_low, Dlow.transpose(), Dlow, Ma_low, Mtinv_low, Slow, M, rhs)
# print('low all: it={}, norm={:.3e}'.format(solver.it, solver.norm))