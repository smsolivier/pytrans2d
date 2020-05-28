#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from trans2d import * 
from OutputCycler import OutputCycler

oc = OutputCycler()

def spcond(A):
	norm = spla.norm(A)
	norminv = spla.norm(spla.inv(A))
	return norm*norminv

def KappaMix(Ne, p):
	mesh = RectMesh(Ne, Ne)
	h1 = H1Space(mesh, LobattoBasis, p, 2)
	l2 = L2Space(mesh, LegendreBasis if p>1 else LegendreBasis, p-1, 1, False)
	h1l = h1.LORefine()
	l2l = L2Space(h1l.mesh, LegendreBasis, 0, 1, False)

	Mt = Assemble(h1, VectorMassIntegrator, lambda x: -1, 2*p+1)
	D = MixAssemble(l2, h1, MixDivIntegrator, 1, 2*p+1)
	DT = D.transpose()

	Mtl = Assemble(h1l, VectorMassIntegrator, lambda x: -1, 2*p+1)
	Dl = MixAssemble(l2l, h1l, MixDivIntegrator, 1, 2*p+1)
	DlT = Dl.transpose()

	M = sp.bmat([[Mt, DT], [D, None]]).tocsc()
	Ml = sp.bmat([[Mtl, DlT], [Dl, None]]).tocsc()

	prec = spla.inv(Ml)*M 
	return np.linalg.cond(prec.todense()), np.linalg.cond(M.todense())

def KappaSecond(Ne, p):
	mesh = RectMesh(Ne, Ne)
	h1 = H1Space(mesh, LobattoBasis, p)
	h1l = h1.LORefine()

	K = Assemble(h1, DiffusionIntegrator, lambda x: 1, 2*p+1).tolil()
	Kl = Assemble(h1l, DiffusionIntegrator, lambda x: 1, 2*p+1).tolil()
	K[h1.bnodes,:] = 0 
	K[h1.bnodes,h1.bnodes] = 1 
	Kl[h1.bnodes,:] = 0 
	Kl[h1.bnodes,h1.bnodes] = 1 

	K = K.tocsc()
	Kl = Kl.tocsc()

	prec = spla.inv(Kl)*K 
	return np.linalg.cond(prec.todense()), np.linalg.cond(K.todense())

Ne = np.array([2, 4, 8, 12])
p = np.array([2,3,4])

for i in range(len(p)):
	print('starting p = {}'.format(p[i]))
	start = time.time()
	mprec = np.zeros(len(Ne))
	morig = np.zeros(len(Ne))
	sprec = np.zeros(len(Ne))
	sorig = np.zeros(len(Ne))
	for j in range(len(Ne)):
		mprec[j], morig[j] = KappaMix(Ne[j], p[i])
		sprec[j], sorig[j] = KappaSecond(Ne[j], p[i]) 

	plt.figure()
	plt.semilogy(Ne**2, mprec, '-o', label='Mixed LOR Prec.')
	plt.semilogy(Ne**2, morig, '-o', label='Mixed HO')
	plt.semilogy(Ne**2, sprec, '-o', label='H1 LOR Prec.')
	plt.semilogy(Ne**2, sorig, '-o', label='H1 HO')
	plt.xlabel('Number of Elements')
	plt.ylabel('Condition Number')
	plt.legend(prop={'size':14})
	if (oc.Good()):
		plt.savefig(oc.Get())

	print('completed in {:.3f} s'.format(time.time() - start))

if not(oc.Good()):
	plt.show()