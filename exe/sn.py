#!/usr/bin/env python3

import numpy as np
import sys

from trans2d import * 

Ne = 10 
p = 1 
N = 2
if (len(sys.argv)>1):
	Ne = int(sys.argv[1])
if (len(sys.argv)>2):
	p = int(sys.argv[2])
if (len(sys.argv)>3):
	N = int(sys.argv[3])

mesh = RectMesh(Ne, Ne)
space = L2Space(mesh, LagrangeBasis, p)
quad = LevelSym(N)

sigma_t = lambda x: 1
sigma_s = lambda x: 0
Q = lambda x, Omega: 0
psi_in = lambda x, Omega: 1
# sweep = Sweeper(space, quad, sigma_t, sigma_s, Q, psi_in, True)
sweep = DirectSweeper(space, quad, sigma_t, sigma_s, Q, psi_in, True)
sn = P1SA(sweep)
psi = TVector(space, quad)
phi = sn.SourceIteration(psi)

# mesh.WriteVTK('solution', cell={'phi':phi.ElementData()})
# mesh.WriteVTK('solution', cell={
	# 'psi1':psi.gf[0].ElementData(), 'psi2':psi.gf[1].ElementData(),
	# 'psi3':psi.gf[2].ElementData(), 'psi4':psi.gf[3].ElementData(), 
	# 'phi':phi.ElementData()})