#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sys 

from trans2d import * 

Ne = 10 
p = 3 
if (len(sys.argv)>1):
	Ne = int(sys.argv[1])
if (len(sys.argv)>2):
	p = int(sys.argv[2]) 

# mesh = RectMesh(Ne, Ne) 
# space = H1Space(mesh, LobattoBasis, p) 
# lospace = space.LORefine()
# print('Nu = {}'.format(space.Nu))

# K = Assemble(space, DiffusionIntegrator, lambda x: 1, 2*p+1)
# Klow = Assemble(lospace, DiffusionIntegrator, lambda x: 1, 2*p+1)
# f = AssembleRHS(space, DomainIntegrator, 
# 	lambda x: 2*np.pi**2*np.sin(np.pi*x[0])*np.sin(np.pi*x[1]), 2*p+1)

# K = K.tolil()
# Klow = Klow.tolil()
# K[space.bnodes,:] = 0 
# K[space.bnodes,space.bnodes] = 1 
# Klow[space.bnodes,:] = 0 
# Klow[space.bnodes,space.bnodes] = 1 
# K = K.tocsc()
# Klow = Klow.tocsc()

# prec = spla.inv(Klow)*K
# print('kappa(prec) = {:.3e}'.format(np.linalg.cond(prec.todense())))
# print('kappa(K) = {:.3e}'.format(np.linalg.cond(K.todense())))

# solver = AMGSolver(1e-10, 1000, 1, False)
# solver.Solve(K, Klow, f)
# print('lor: it={}, norm={:.3e}'.format(solver.it, solver.norm))
# solver.Solve(K, K, f)
# print('amg: it={}, norm={:.3e}'.format(solver.it, solver.norm))

mesh = RectMesh(Ne, Ne)
h1 = H1Space(mesh, LobattoBasis, p, 2)
l2 = L2Space(mesh, LegendreBasis if p>1 else LegendreBasis, p-1, 1, False)
h1l = h1.LORefine()
l2l = L2Space(h1l.mesh, LegendreBasis, 0, 1, False)
print('Nu = {}'.format(h1.Nu + l2.Nu))
l2.Plot()
l2l.Plot()
h1.Plot()
h1l.Plot()
plt.show()

Mt = Assemble(h1, VectorMassIntegrator, lambda x: 1, 2*p+1)
MtLump = Assemble(h1, VectorMassIntegratorRowSum, lambda x: 1, 2*p+1)
MtLumpInv = sp.diags(1/MtLump.diagonal())
Ma = Assemble(l2, MassIntegrator, lambda x: .1, 2*p+1)
D = MixAssemble(l2, h1, MixDivIntegrator, 1, 2*p+1)
DT = D.transpose()
f = AssembleRHS(l2, DomainIntegrator, 
	lambda x: (2*np.pi**2+.1)*np.sin(np.pi*x[0])*np.sin(np.pi*x[1]), 2*p+1)

M = sp.bmat([[Mt, -DT], [D, Ma]]).tocsc()
rhs = np.concatenate((np.zeros(h1.Nu), f))
S = Ma + D*MtLumpInv*DT 

Mtl = Assemble(h1l, VectorMassIntegrator, lambda x: 1, 2*p+1)
MtlLump = Assemble(h1l, VectorMassIntegratorRowSum, lambda x: 1, 2*p+1)
MtlLumpInv = sp.diags(1/MtlLump.diagonal())
Mal = Assemble(l2l, MassIntegrator, lambda x: .1, 2*p+1)
Dl = MixAssemble(l2l, h1l, MixDivIntegrator, 1, 2*p+1)
DlT = Dl.transpose()
fl = AssembleRHS(l2l, DomainIntegrator, 
	lambda x: (2*np.pi**2+.1)*np.sin(np.pi*x[0])*np.sin(np.pi*x[1]), 2*p+1)

Ml = sp.bmat([[Mtl, -DlT], [Dl, Mal]]).tocsc()
rhsl = np.concatenate((np.zeros(h1.Nu), fl))
Sl = Mal + Dl*MtLumpInv*Dl.transpose()

# Shat = Mal + Dl*MtlLumpInv*DlT
# Shat = Ma + D*MtLumpInv*DT 
# L = sp.bmat([[sp.identity(h1.Nu), None], [D*MtLumpInv, sp.identity(l2.Nu)]])
# D = sp.bmat([[Mt, None], [None, Shat]])
# U = sp.bmat([[sp.identity(h1.Nu), -MtLumpInv*DT], [None, sp.identity(l2.Nu)]])
# P = (L*D*U).tocsc()

prec = spla.inv(Sl)*S 
print('kappa(prec) = {:.3f}'.format(np.linalg.cond(prec.todense())))
print('kappa(S) = {:.3f}'.format(np.linalg.cond(S.todense())))

# prec = spla.inv(Ml)*M
# print('kappa(prec) = {:.3f}'.format(np.linalg.cond(prec.todense())))
# print('kappa(M) = {:.3f}'.format(np.linalg.cond(M.todense())))
# print('kappa(Ml) = {:.3f}'.format(np.linalg.cond(Ml.todense())))

# prec = spla.inv(Sl)*S 
# print('kappa(Sprec) = {:.3f}'.format(np.linalg.cond(prec.todense())))
# print('kappa(S) = {:.3f}'.format(np.linalg.cond(S.todense())))

# x = spla.spsolve(M, rhs)
# T = GridFunction(l2)
# T.data = x[h1.Nu:]

# xl = spla.spsolve(Ml, rhsl)
# Tl = GridFunction(l2l)
# Tl.data = x[h1.Nu:]

# err = T.L2Error(lambda x: np.sin(np.pi*x[0])*np.sin(np.pi*x[1]), 2*p+2)
# print('err = {:.3e}'.format(err))

# T.space.mesh.WriteVTK('solution', cell={'T':T.ElementData()})