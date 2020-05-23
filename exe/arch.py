#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from trans2d import * 

Nt = 20
Nr = 4
p = 2 
mp = 1
if (len(sys.argv)>1):
	Nt = int(sys.argv[1])
if (len(sys.argv)>2):
	Nr = int(sys.argv[2])
if (len(sys.argv)>3):
	p = int(sys.argv[3])
if (len(sys.argv)>4):
	mp = int(sys.argv[4])

ri = .5
ro = 1
theta = np.linspace(0, np.pi, Nt+1)
R = np.linspace(ri, ro, Nr+1)
Nn = len(theta)*len(R)
Ne = (Nt)*Nr 

nodes = np.zeros((Nn, 2))
nodes[:,0] = np.outer(R, np.cos(theta)).flatten()
nodes[:,1] = np.outer(R, np.sin(theta)).flatten()

ele = np.zeros((Ne, 4), dtype=int) 
e = 0 
for r in range(Nr):
	for t in range(Nt):
		ele[e,0] = t + r*(Nt+1) + 1
		ele[e,1] = t + r*(Nt+1) 
		ele[e,2] = t + (r+1)*(Nt+1) + 1 
		ele[e,3] = t + (r+1)*(Nt+1)  
		e += 1 	

# for r in range(Nr):
# 	t = Nt-1
# 	ele[e,1] = Nt + r*(Nt+1) 
# 	ele[e,3] = Nt + (r+1)*(Nt+1)
# 	ele[e,0] = r*(Nt+1)
# 	ele[e,2] = (r+1)*(Nt+1)
# 	e += 1 
mesh = AbstractMesh(nodes, ele, mp)
for e in range(Ne):
	area = mesh.trans[e].Area()
	assert(area>0)

rt = RTSpace(mesh, LobattoBasis, LegendreBasis, p)
l2 = L2Space(mesh, LegendreBasis, p) 
print('rt Nu = {}\nl2 Nu = {}\ntotal Nu = {}'.format(rt.Nu, l2.Nu, rt.Nu+l2.Nu))

qorder = 2*(p+1)+1
M = Assemble(rt, VectorFEMassIntegrator, lambda x: -1, qorder)
D = MixAssemble(l2, rt, VectorFEDivIntegrator, 1, qorder) 
Q = lambda x: 1 
f = AssembleRHS(l2, DomainIntegrator, Q, qorder) 

A = sp.bmat([[M, D.transpose()], [D, None]]).tocsc()
rhs = np.concatenate((np.zeros(rt.Nu), f))
x = spla.spsolve(A, rhs) 

T = GridFunction(l2)
T.data = x[rt.Nu:]
q = GridFunction(rt)
q.data = x[:rt.Nu]

mesh.WriteVTK('solution', cell={'T':T.ElementData(), 'q':q.ElementData()})

# h1 = H1Space(mesh, LobattoBasis, p)
# print('Nu = {}'.format(h1.Nu))
# K = Assemble(h1, DiffusionIntegrator, lambda x: 1, 2*p+1).tolil()
# K[h1.bnodes,:] = 0 
# K[h1.bnodes, h1.bnodes] = 1 
# K = K.tocsc()

# f = AssembleRHS(h1, DomainIntegrator, lambda x: 1, 2*p+1)
# T = GridFunction(h1)
# T.data = spla.spsolve(K, f) 

# mesh.WriteVTK('solution', cell={'T':T.ElementData()})

# space = L2Space(mesh, LegendreBasis, p)
# N = 4
# quad = LevelSym(N)
# print('Nu = {}'.format(space.Nu*quad.N))

# sigma_t = lambda x: 1
# sigma_s = lambda x: 1
# Q = lambda x, Omega: 0
# def psi_in(x, Omega):
# 	r = np.sqrt(x[0]**2 + x[1]**2) 
# 	# if (r < ri*1.1 and x[0]>-.25 and x[0]<.25):
# 	if ((x[0]<-ri or x[0]>ri) and x[1]<1e-5):
# 		return 1
# 	else:
# 		return 0 
# # psi_in = lambda x, Omega: 0
# # sweep = Sweeper(space, quad, sigma_t, sigma_s, Q, psi_in, True)
# sweep = DirectSweeper(space, quad, sigma_t, sigma_s, Q, psi_in, True)
# sn = P1SA(sweep)
# psi = TVector(space, quad)
# phi = sn.SourceIteration(psi)

# mesh.WriteVTK('solution', cell={'phi':phi.ElementData()})