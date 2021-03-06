#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from trans2d import * 
import pytest 

trans = AffineTrans(np.array([[0,0], [1,0], [0,1], [1,1]]))
el = Element(LagrangeBasis, 1)
el0 = Element(LegendreBasis, 0)

def test_diffusion():
	K = DiffusionIntegrator(el, trans, lambda x: 1, 2)
	Kex = np.array([[2/3, -1/6, -1/6, -1/3],
		[-1/6, 2/3, -1/3, -1/6], 
		[-1/6, -1/3, 2/3, -1/6], 
		[-1/3, -1/6, -1/6, 2/3]])
	assert(K==pytest.approx(Kex))

def test_mass():
	M = MassIntegrator(el, trans, lambda x: 1, 2)
	Mex = np.array([[1/9, 1/18, 1/18, 1/36], 
		[1/18, 1/9, 1/36, 1/18], 
		[1/18, 1/36, 1/9, 1/18], 
		[1/36, 1/18, 1/18, 1/9]])
	assert(M==pytest.approx(Mex))

def test_mass_on_quad():
	trans = ElementTrans(np.array([[0,0], [1,0], [-.25,1], [1.25,1]]))
	M = MassIntegrator(el, trans, lambda x: 1, 3)
	Mex = np.array([[1/8, 1/16, 5/72, 5/144], 
		[1/16, 1/8, 5/144, 5/72], 
		[5/72, 5/144, 11/72, 11/144], 
		[5/144, 5/72, 11/144, 11/72]])
	assert(M==pytest.approx(Mex))

def test_mixmass():
	M = MixMassIntegrator(el, el0, trans, lambda x: 1, 2)
	Mex = np.ones((4,1))/4
	assert(M==pytest.approx(Mex))

def test_vecmass():
	M = VectorMassIntegrator(el, trans, lambda x: 1, 2)

	Ms = np.array([[1/9, 1/18, 1/18, 1/36], 
		[1/18, 1/9, 1/36, 1/18], 
		[1/18, 1/36, 1/9, 1/18], 
		[1/36, 1/18, 1/18, 1/9]])

	Mex = np.block([[Ms, np.zeros(Ms.shape)], [np.zeros(Ms.shape), Ms]])
	assert(M==pytest.approx(Mex))

def test_assemble():
	mesh = RectMesh(3,3)
	space = H1Space(mesh, LagrangeBasis, 1)
	K = Assemble(space, DiffusionIntegrator, lambda x: 1, 0)

	assert(spla.norm(K - K.transpose())==pytest.approx(0))

def test_wci():
	cx = 1 
	cy = 1 
	W = WeakConvectionIntegrator(el, trans, [cx,cy], 2)
	Wex = 1/12*np.array([[2*(cx+cy), 2*cx+cy, cx+2*cy, cx+cy], 
		[-2*cx+cy, 2*(-cx+cy), -cx+cy, -cx+2*cy], 
		[cx-2*cy, cx-cy, 2*(cx-cy), 2*cx-cy], 
		[-cx-cy, -cx-2*cy, -2*cx-cy, -2*(cx+cy)]])
	assert(W==pytest.approx(Wex))

def test_mixdiv():
	D = MixDivIntegrator(el, el, trans, 1, 2)
	Dex = np.array([
		[-1/6, 1/6, -1/12, 1/12, -1/6, -1/12, 1/6, 1/12], 
		[-1/6, 1/6, -1/12, 1/12, -1/12, -1/6, 1/12, 1/6], 
		[-1/12, 1/12, -1/6, 1/6, -1/6, -1/12, 1/6, 1/12], 
		[-1/12, 1/12, -1/6, 1/6, -1/12, -1/6, 1/12, 1/6]])
	assert(D==pytest.approx(Dex))

def test_weakmixdiv():
	D = WeakMixDivIntegrator(el, el, trans, 1, 2)
	Dex = np.array([[-1/6, -1/6, -1/12, -1/12, -1/6, -1/12, -1/6, -1/12], 
		[1/6, 1/6, 1/12, 1/12, -1/12, -1/6, -1/12, -1/6], 
		[-1/12, -1/12, -1/6, -1/6, 1/6, 1/12, 1/6, 1/12], 
		[1/12, 1/12, 1/6, 1/6, 1/12, 1/6, 1/12, 1/6]])
	assert(D==pytest.approx(-Dex))

def test_face():
	mesh = RectMesh(3,3, [0,0], [3,3])
	fi = mesh.iface[0]
	face = fi.face 

	ip = .25
	xi1 = fi.ipt1.Transform(ip)
	xi2 = fi.ipt2.Transform(ip)
	X1 = fi.trans1.Transform(xi1)
	X2 = fi.trans2.Transform(xi2)
	assert(X1==pytest.approx(X2))

	J = JumpJumpIntegrator(el, el, fi, .25, 3)
	Jex = np.array([[
		0,0,0,0,0,0,0,0],
		[0,1/3,0,1/6,-(1/3),0,-(1/6),0],
		[0,0,0,0,0,0,0,0],
		[0,1/6,0,1/3,-(1/6),0,-(1/3),0],
		[0,-(1/3),0,-(1/6),1/3,0,1/6,0],
		[0,0,0,0,0,0,0,0],
		[0,-(1/6),0,-(1/3),1/6,0,1/3,0],
		[0,0,0,0,0,0,0,0]])

	assert(J==pytest.approx(.25*Jex))

	jva = MixJumpVAvgIntegrator([el,el], [el,el], fi, 1, 3)
	jva_ex = np.array([
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
		[0, 1/6, 0, 1/12, 0, 0, 0, 0, 1/6, 0, 1/12, 0, 0, 0, 0, 0], 
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
		[0, 1/12, 0, 1/6, 0, 0, 0, 0, 1/12, 0, 1/6, 0, 0, 0, 0, 0], 
		[0, -(1/6), 0, -(1/12), 0, 0, 0, 0, -(1/6), 0, -(1/12), 0, 0, 0, 0, 0], 
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
		[0, -(1/12), 0, -(1/6), 0, 0, 0, 0, -(1/12), 0, -(1/6), 0, 0, 0, 0, 0], 
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		])
	assert(jva==pytest.approx(jva_ex))

	vj = VectorJumpJumpIntegrator(el, el, fi, 1, 3)
	vj_ex = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
		[0,1/3,0,1/6,0,0,0,0,-(1/3),0,-(1/6),0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
		[0,1/6,0,1/3,0,0,0,0,-(1/6),0,-(1/3),0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,-(1/3),0,-(1/6),0,0,0,0,1/3,0,1/6,0,0,0,0,0],
		[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,-(1/6),0,-(1/3),0,0,0,0,1/6,0,1/3,0,0,0,0,0],
		[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
	assert(vj==pytest.approx(vj_ex))

	vja = VectorJumpAvgIntegrator([el,el], [el,el], fi, 1, 3)
	vja_ex = np.array([[0, 0, 0, 0, 0, 0, 0, 0], [0, 1/6, 0, 1/12, 1/6, 0, 1/12, 0], 
		[0, 0, 0, 0, 0, 0, 0, 0], [0, 1/12, 0, 1/6, 1/12, 0, 1/6, 0], 
		[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], 
		[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], 
		[0, -(1/6), 0, -(1/12), -(1/6), 0, -(1/12), 0], [0, 0, 0, 0, 0, 0, 0, 0], 
		[0, -(1/12), 0, -(1/6), -(1/12), 0, -(1/6), 0], [0, 0, 0, 0, 0, 0, 0, 0], 
		[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], 
		[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]])
	assert(vja==pytest.approx(vja_ex))

	fi = mesh.iface[1]
	jj = JumpJumpIntegrator(el, el, fi, .25, 3)
	jj_ex = np.array([[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],
		[0,0,1/3,1/6,-(1/3),-(1/6),0,0],[0,0,1/6,1/3,-(1/6),-(1/3),0,0],
		[0,0,-(1/3),-(1/6),1/3,1/6,0,0],[0,0,-(1/6),-(1/3),1/6,1/3,0,0],
		[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]])
	assert(jj==pytest.approx(.25*jj_ex))

	jva = MixJumpVAvgIntegrator([el,el], [el,el], fi, 1, 3)
	jva_ex = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,1/6,1/12,0,0,0,0,1/6,1/12,0,0],[0,0,0,0,0,0,1/12,1/6,0,0,0,0,1/12,1/6,0,0],
		[0,0,0,0,0,0,-(1/6),-(1/12),0,0,0,0,-(1/6),-(1/12),0,0],[0,0,0,0,0,0,-(1/12),-(1/6),0,0,0,0,-(1/12),-(1/6),0,0],
		[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
	assert(jva==pytest.approx(jva_ex))

	vjj = VectorJumpJumpIntegrator(el, el, fi, 1, 3)
	vjj_ex = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,1/3,1/6,0,0,0,0,-(1/3),-(1/6),0,0],[0,0,0,0,0,0,1/6,1/3,0,0,0,0,-(1/6),-(1/3),0,0],
		[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,-(1/3),-(1/6),0,0,0,0,1/3,1/6,0,0],[0,0,0,0,0,0,-(1/6),-(1/3),0,0,0,0,1/6,1/3,0,0],
		[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
	assert(vjj==pytest.approx(vjj_ex))

	vja = VectorJumpAvgIntegrator([el,el], [el,el], fi, 1, 3)
	vja_ex = np.array([[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],
		[0,0,1/6,1/12,1/6,1/12,0,0],[0,0,1/12,1/6,1/12,1/6,0,0],[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],
		[0,0,-(1/6),-(1/12),-(1/6),-(1/12),0,0],[0,0,-(1/12),-(1/6),-(1/12),-(1/6),0,0],
		[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]])
	assert(vja==pytest.approx(vja_ex))

def test_block():
	mesh = RectMesh(3,3)
	space = H1Space(mesh, LobattoBasis, 2, 2)
	Mt1 = AssembleBlocks(space, VectorMassIntegrator, lambda x: 1, 5)
	Mt2 = AssembleBlocks(space, VectorMassIntegrator, lambda x: 2, 5) 
	Mt3 = Mt1 + Mt2 
	Mt1 += Mt2 
	for i in range(2):
		for j in range(2):
			assert(spla.norm(Mt1[i,j] - Mt3[i,j])<1e-15)

def test_rt0div():
	rt = RTElement(LobattoBasis, LegendreBasis, 0) 
	rt.modal = True
	l2 = Element(LegendreBasis, 0) 

	D = VectorFEDivIntegrator(l2, rt, trans, 1, 3) 
	Dex = np.array([[-1,1,-1,1]])
	assert(D==pytest.approx(Dex))

def test_rt0mass():
	rt = RTElement(LobattoBasis, LegendreBasis, 0) 
	rt.modal = True
	M = VectorFEMassIntegrator(rt, trans, lambda x: 1, 3) 
	Mex = np.array([[1/3, 1/6, 0, 0], [1/6, 1/3, 0, 0], [0, 0, 1/3, 1/6], [0, 0, 1/6, 1/3]])
	assert(M==pytest.approx(Mex))
