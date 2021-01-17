#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sys 

from trans2d import * 
from OutputCycler import OutputCycler

oc = OutputCycler()
p = oc.GetOpt(0, 2)
sm_inner = oc.GetOpt(1, 1)

Ne = 10 
if (len(sys.argv)>1):
	Ne = int(sys.argv[1])
if (len(sys.argv)>2):
	p = int(sys.argv[2]) 

mesh = RectMesh(Ne, Ne) 
space = H1Space(mesh, LobattoBasis, p) 
lospace = space.LORefine()
print('Nu = {}'.format(space.Nu))

K = Assemble(space, DiffusionIntegrator, lambda x: 1, 2*p+1)
Klow = Assemble(lospace, DiffusionIntegrator, lambda x: 1, 2*p+1)
f = AssembleRHS(space, DomainIntegrator, 
	lambda x: 2*np.pi**2*np.sin(np.pi*x[0])*np.sin(np.pi*x[1]), 2*p+1)

K = K.tolil()
Klow = Klow.tolil()
K[space.bnodes,:] = 0 
K[space.bnodes,space.bnodes] = 1 
Klow[space.bnodes,:] = 0 
Klow[space.bnodes,space.bnodes] = 1 
K = K.tocsc()
Klow = Klow.tocsc()

# prec = spla.inv(Klow)*K
# print('kappa(prec) = {:.3e}'.format(np.linalg.cond(prec.todense())))
# print('kappa(K) = {:.3e}'.format(np.linalg.cond(K.todense())))

solver = AMGSolver(1e-10, 1000, 1, ('gauss_seidel', {'sweep':'symmetric', 'iterations':1}), False)
# solver = LUSolver(1e-10, 1000, None, False)
solver.Solve(K, Klow, f)
print('lor: it={}, norm={:.3e}'.format(solver.it, solver.norm))
solver.Solve(K, K, f)
print('amg: it={}, norm={:.3e}'.format(solver.it, solver.norm))

# Ne = np.array([4, 8, 12, 16, 20])
# it = np.zeros((4,len(Ne)))

# maxiter = 500
# amg_inner = 1
# for i in range(len(Ne)):
# 	mesh = RectMesh(Ne[i], Ne[i])
# 	h1 = H1Space(mesh, LobattoBasis, p, 2)
# 	l2 = L2Space(mesh, LegendreBasis if p>1 else LegendreBasis, p-1, 1, False)
# 	h1l = h1.LORefine()
# 	l2l = L2Space(h1l.mesh, LegendreBasis, 0, 1, False)
# 	print('Nu = {}'.format(h1.Nu + l2.Nu))
# 	# l2.Plot()
# 	# l2l.Plot()
# 	# h1.Plot()
# 	# h1l.Plot()
# 	# plt.show()

# 	bnodes = []
# 	for bn in h1.bnodes:
# 		if (h1.nodes[bn,1]==1 or h1.nodes[bn,1]<1e-3):
# 			bnodes.append(int(h1.Nu/2) + bn) 

# 	sigt = lambda x: (x[0]>=.5)*100 + (x[0]<.5)*1 
# 	siga = lambda x: (x[0]>=.5)*1 + (x[0]<.5)*.1 
# 	Mt = Assemble(h1, VectorMassIntegrator, sigt, 2*p+1)
# 	MtLump = Assemble(h1, VectorMassIntegratorRowSum, sigt, 2*p+1)
# 	MtLumpInv = sp.diags(1/MtLump.diagonal())
# 	Ma = Assemble(l2, MassIntegrator, siga, 2*p+1)
# 	D = MixAssemble(l2, h1, MixDivIntegrator, 1, 2*p+1)
# 	DT = D.transpose()
# 	# f = AssembleRHS(l2, DomainIntegrator, 
# 		# lambda x: (2*np.pi**2+.1)*np.sin(np.pi*x[0])*np.sin(np.pi*x[1]), 2*p+1)
# 	# f = AssembleRHS(l2, DomainIntegrator, lambda x: (np.pi**2 + .1)*np.sin(np.pi*x[0]), 2*p+1)
# 	f = AssembleRHS(l2, DomainIntegrator, lambda x: 1, 2*p+1)

# 	M = sp.bmat([[Mt, -DT], [D, Ma]]).tocsc()
# 	rhs = np.concatenate((np.zeros(h1.Nu), f))
# 	S = Ma + D*MtLumpInv*DT 

# 	Mtl = Assemble(h1l, VectorMassIntegrator, sigt, 2*p+1)
# 	MtlLump = Assemble(h1l, VectorMassIntegratorRowSum, sigt, 2*p+1)
# 	MtlLumpInv = sp.diags(1/MtlLump.diagonal())
# 	Mal = Assemble(l2l, MassIntegrator, siga, 2*p+1)
# 	Dl = MixAssemble(l2l, h1l, MixDivIntegrator, 1, 2*p+1)
# 	DlT = Dl.transpose()

# 	Ml = sp.bmat([[Mtl, -DlT], [Dl, Mal]]).tocsc()
# 	Sl = Mal + Dl*MtLumpInv*Dl.transpose()

# 	smoother = SymGaussSeidel(1e-10, sm_inner, False)
# 	solver = AMGSolver(1e-10, maxiter, amg_inner, smoother, False)
# 	solver.Solve(S, Sl, f)
# 	print('sgs = {}'.format(solver.it))
# 	it[0,i] = solver.it 

# 	smoother = GaussSeidel(1e-10, sm_inner, False)
# 	solver = AMGSolver(1e-10, maxiter, amg_inner, smoother, False)
# 	solver.Solve(S, Sl, f)
# 	print('gs = {}'.format(solver.it))
# 	it[1,i] = solver.it 

# 	smoother = Jacobi(1e-10, sm_inner, False)
# 	solver = AMGSolver(1e-10, maxiter, amg_inner, smoother, False)
# 	solver.Solve(S, Sl, f)
# 	print('jacobi = {}'.format(solver.it))
# 	it[2,i] = solver.it 

# 	solver = AMGSolver(1e-10, maxiter, amg_inner, None, False)
# 	solver.Solve(S, Sl, f)
# 	print('none = {}'.format(solver.it))
# 	it[3,i] = solver.it 

# plt.plot(Ne**2, it[3], '-o', label='None')
# plt.plot(Ne**2, it[2], '-o', label='Jacobi')
# plt.plot(Ne**2, it[1], '-o', label='GS')
# plt.plot(Ne**2, it[0], '-o', label='Sym. GS')
# plt.xlabel('Number of Elements')
# plt.ylabel('GMRES Iterations')
# plt.title(r'$p='+str(p)+'$, smoother it=' + str(sm_inner))
# plt.legend()
# if (oc.Good()):
# 	plt.savefig(oc.Get())
# else:
# 	plt.show()