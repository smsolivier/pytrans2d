#!/usr/bin/env python3

from trans2d import * 

Ne = 10 
p = 1 
N = 4
if (len(sys.argv)>1):
	Ne = int(sys.argv[1])
if (len(sys.argv)>2):
	p = int(sys.argv[2])
if (len(sys.argv)>3):
	N = int(sys.argv[3])

mesh = RectMesh(Ne, Ne)
quad = LevelSym(N)
space = L2Space(mesh, LegendreBasis, p)
phi_space = L2Space(mesh, LegendreBasis, p-1)
J_space = H1Space(mesh, LobattoBasis, p, 2)
print('sn unknowns = {}'.format(space.Nu*quad.N))
print('mip unknowns = {}'.format(space.Nu))
print('p1 unknowns = {}'.format(3*space.Nu))
print('vef unknowns = {}'.format(phi_space.Nu + J_space.Nu))
eps = 1e-3
c = .99
# sigma_t = lambda x: 1/eps
# sigma_s = lambda x: 1/eps - eps
# Q = lambda x, Omega: eps
def sigma_t(X):
	x = X[0]
	y = X[1] 
	if (x>.4 and x<.6 and y>.4 and y<.6):
		return 1/eps
	else:
		return 1
def sigma_s(X):
	x = X[0]
	y = X[1]
	if (x>.4 and x<.6 and y>.4 and y<.6):
		return 1/eps - eps 
	else:
		return 1
def Q(X,Omega):
	x = X[0]
	y = X[1]
	if (x>.4 and x<.6 and y>.4 and y<.6):
		return 0
	else:
		return 1 
psi_in = lambda x, Omega: 0

sweep = DirectSweeper(space, quad, sigma_t, sigma_s, Q, psi_in)
# sn = MIP(sweep)
sn = P1SA(sweep)
psi = TVector(space, quad)
phi_sn = sn.SourceIteration(psi)
solver = BlockLDU(1e-10, 1000, 1, False)
vef = VEF(phi_space, J_space, sweep, solver)
# vef.full_lump = True
phi, J = vef.Mult(psi)
print('gmres: norm={:.3e}, it={}'.format(solver.norm, solver.it))
mesh.WriteVTK('solution', cell={'phi':phi.ElementData(), 'J':J.ElementData(), 'phi_sn':phi_sn.ElementData()})
