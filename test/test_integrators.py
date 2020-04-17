#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from trans2d import * 
import pytest 

trans = AffineTrans(np.array([[0,0], [1,0], [1,1], [0,1]]))
el = Element(LagrangeBasis, 1)

def test_diffusion():
	K = DiffusionIntegrator(el, trans, lambda x: 1, 0)
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
	trans = LinearTrans(np.array([[0,0], [1,0], [-.25,1], [1.25,1]]))
	M = MassIntegrator(el, trans, lambda x: 1, 3)
	Mex = np.array([[1/8, 1/16, 5/72, 5/144], 
		[1/16, 1/8, 5/144, 5/72], 
		[5/72, 5/144, 11/72, 11/144], 
		[5/144, 5/72, 11/144, 11/72]])
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