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
start = time.time()
space = H1Space(mesh, LagrangeBasis, p)
lospace = space.LORefine()
print('space time = {:.3f} s'.format(time.time() - start))
print('Nu = {}'.format(space.Nu))

start = time.time()
# kappa = lambda x: 1e-3*(x[0]>.5) + 1*(x[0]<=.5)
kappa = lambda x: 1 
# kappa = lambda x: 1 
K = Assemble(space, DiffusionIntegrator, kappa, 2*p+2)
# Klow = Assemble(lospace, DiffusionIntegrator, kappa, 2*p+1)
Klow = AssembleLOR(space, DiffusionIntegrator, kappa, 2*p+1)
# f = AssembleRHS(space, DomainIntegrator, lambda x: 2*np.pi**2*np.sin(np.pi*x[0])*np.sin(np.pi*x[1]), 2*p+1)
f = AssembleRHS(space, DomainIntegrator, lambda x: 1, 2*p+1)
print('assembly time = {:.3f} s'.format(time.time() - start))

start = time.time()
K = K.tolil()
K[space.bnodes,:] = 0 
K[space.bnodes,space.bnodes] = 1
f[space.bnodes] = 0 
Klow = Klow.tolil()
Klow[space.bnodes,:] = 0 
Klow[space.bnodes,space.bnodes] = 1
print('bc time = {:.3f} s'.format(time.time() - start))

start = time.time()
T = GridFunction(space)
amg = pyamg.ruge_stuben_solver(Klow.tocsr())
r = []
T.data = amg.solve(f, maxiter=500, residuals=r)
print('it={}'.format(len(r)))

amg = pyamg.ruge_stuben_solver(K.tocsr())
r = []
T.data = amg.solve(f, maxiter=500, residuals=r)
print('it={}'.format(len(r)))

solver = AMGSolver(1e-10, 500, 1, False)
T.data = solver.Solve(K, K, f)
print('amg: it={}, norm={:.3e}'.format(solver.it, solver.norm))
T.data = solver.Solve(K, Klow, f)
print('low: it={}, norm={:.3e}'.format(solver.it, solver.norm))
print('solve time = {:.3f} s'.format(time.time() - start))

# err = T.L2Error(lambda x: np.sin(np.pi*x[0])*np.sin(np.pi*x[1]), 2*p)
# print('err = {:.3e}'.format(err))

Tex = GridFunction(space)
Tex.Project(lambda x: np.sin(np.pi*x[0])*np.sin(np.pi*x[1]))
mesh.WriteVTK('solution', cell={'T':T.ElementData(), 'Tex':Tex.ElementData()})