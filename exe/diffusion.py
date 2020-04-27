#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sys 
import time

from trans2d import * 

N = 4
p = 1 
if (len(sys.argv)>1):
	N = int(sys.argv[1])
if (len(sys.argv)>2):
	p = int(sys.argv[2])
start = time.time()
mesh = RectMesh(N, N)
print('mesh time = {:.3f} s'.format(time.time() - start))
space = H1Space(mesh, LagrangeBasis, p)
lospace = space.LORefine()
start = time.time()
print('space time = {:.3f} s'.format(time.time() - start))
print('Nu = {}'.format(space.Nu))

start = time.time()
K = Assemble(space, DiffusionIntegrator, lambda x: 1, 2*p+1)
Klow = Assemble(lospace, DiffusionIntegrator, lambda x: 1, 2*p+1)
f = AssembleRHS(space, DomainIntegrator, lambda x: 2*np.pi**2*np.sin(np.pi*x[0])*np.sin(np.pi*x[1]), 2*p+1)
print('assembly time = {:.3f} s'.format(time.time() - start))

start = time.time()
K = K.tolil()
K[space.bnodes,:] = 0 
K[space.bnodes,space.bnodes] = 1
f[space.bnodes] = 0 
Klow = Klow.tolil()
Klow[lospace.bnodes,:] = 0 
Klow[lospace.bnodes,lospace.bnodes] = 1
print('bc time = {:.3f} s'.format(time.time() - start))

start = time.time()
T = GridFunction(space)
# T.data = spla.spsolve(K.tocsc(), f)
amglo = pyamg.ruge_stuben_solver(Klow.tocsr())
amg = pyamg.ruge_stuben_solver(K.tocsr())
it = 0
def cb(r):
	global it 
	norm = np.linalg.norm(r)
	it += 1 
	print('i={:3}, norm={:.3e}'.format(it, norm))
T.data, info = spla.gmres(K.tocsc(), f, M=amglo.aspreconditioner(cycle='V'), callback=cb, callback_type='legacy', tol=1e-10, atol=1e-10, maxiter=1000, restart=None)
# T.data, info = spla.gmres(K.tocsc(), f, M=amg.aspreconditioner(cycle='V'), callback=cb, callback_type='legacy', tol=1e-10, atol=0, maxiter=1000, restart=None)
if (info!=0):
	print('gmres issue :(')
print('solve time = {:.3f} s'.format(time.time() - start))

err = T.L2Error(lambda x: np.sin(np.pi*x[0])*np.sin(np.pi*x[1]), 2*p)
print('err = {:.3e}'.format(err))

Tex = GridFunction(space)
Tex.Project(lambda x: np.sin(np.pi*x[0])*np.sin(np.pi*x[1]))
mesh.WriteVTK('solution', cell={'T':T.ElementData(), 'Tex':Tex.ElementData()})