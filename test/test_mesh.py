#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from trans2d import * 
from pytest import approx 

mesh = RectMesh(3, 3)

def test_trans():
	trans = mesh.trans[0]
	assert(trans.Transform([-1,-1])==approx([0,0]))
	assert(trans.Transform([1,1])==approx([1/3,1/3]))
	assert(trans.Transform([0,-1])==approx([1/6,0]))

def test_jacobian():
	trans = mesh.trans[0]
	assert(trans.Jacobian([0,0])==approx(1/36))

def test_boundary():
	assert(mesh.bel==[0,1,2,3,5,6,7,8])

def test_inverse():
	trans = mesh.trans[0]
	assert(trans.InverseMap([1/6, 1/6])==approx([0,0]))
	assert(trans.InverseMap([1/3,1/3])==approx([1,1]))
	assert(trans.InverseMap([1/3,1/6])==approx([1,0]))

def test_facetrans():
	ft = mesh.iface[0].face
	assert(ft.Transform(0)==approx([1/3,1/6]))
	assert(ft.Jacobian(0)==approx(1/6))
	assert(ft.Normal(0)==approx([1,0]))
	ip, w = quadrature.Get1D(1)
	l = 0
	for n in range(len(w)):
		l += ft.Jacobian(ip[n])*w[n]
	assert(l==approx(1/3))

	ipt1 = mesh.iface[0].ipt1 
	ipt2 = mesh.iface[0].ipt2 
	assert(ipt1.Transform(.5)==approx([1,.5]))
	assert(ipt2.Transform(.5)==approx([-1,.5]))
	assert(mesh.iface[0].fno==0)

def test_bdrface():
	bfi = mesh.bface[0] 
	face = bfi.face
	assert(face.Transform(.5)==approx([1/4,0]))
	assert(bfi.ipt1.Transform(.5)==approx([.5,-1]))

def test_lintrans():
	trans = LinearTrans(np.array([[0,0], [1,0], [1.25,1], [-.25,1]]))
	area = 0 
	ip, w = quadrature.Get(2)
	for n in range(len(w)):
		area += trans.Jacobian(ip[n]) 
	assert(area==approx(5/4))

	xi = trans.InverseMap([859/800, 11/20])
	assert(xi==approx([.9, .1]))
