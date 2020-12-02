#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from trans2d import * 
from pytest import approx 
import pytest

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

	ft = mesh.iface[1].face
	assert(ft.Transform(0)==approx([1/6,1/3]))
	assert(ft.Jacobian(0)==approx(1/6))
	assert(ft.Normal(0)==approx([0,1]))
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

	ipt1 = mesh.iface[1].ipt1 
	ipt2 = mesh.iface[1].ipt2 
	assert(ipt1.Transform(.5)==approx([-.5,1]))
	assert(ipt2.Transform(.5)==approx([-.5,-1]))
	assert(mesh.iface[0].fno==0)

def test_bdrface():
	bfi = mesh.bface[0] 
	face = bfi.face
	assert(face.Transform(.5)==approx([1/4,0]))
	assert(bfi.ipt1.Transform(.5)==approx([.5,-1]))

@pytest.mark.parametrize('TransType', [AffineTrans, ElementTrans])
def test_rotate(TransType):
	box = np.array([[1.,-1], [1,1], [-1,-1], [-1,1]])
	trans = TransType(box)
	assert(trans.Jacobian([0.,0.])==approx(1.))
	assert(trans.Transform([.25,0])==approx([0,.25]))

def test_lintrans():
	trans = ElementTrans(np.array([[0,0], [1,0], [-.25,1], [1.25,1]]))
	area = 0 
	ip, w = quadrature.Get(2)
	for n in range(len(w)):
		area += trans.Jacobian(ip[n]) 
	assert(area==approx(5/4))

	xi = trans.InverseMap([859/800, 11/20])
	assert(xi==approx([.9, .1]))

def test_diamond():
	box = np.array([[0,0], [1,0], [0,1], [1,1]])
	theta = np.pi/4
	R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
	box = np.dot(box, R.transpose()) 
	trans = ElementTrans(box) 
	area = 0 
	ip, w = quadrature.Get(2)
	for n in range(len(w)):
		area += trans.Jacobian(ip[n]) 
	assert(area==approx(1))

	assert(trans.Transform([-1.,-1.])==approx([0,0]))
	assert(trans.Transform([0., 0.])==approx([0, np.sqrt(2)/2]))

@pytest.mark.parametrize('TransType', [AffineTrans, ElementTrans])
def test_intersect(TransType):
	box = np.array([[1.,-1], [1,1], [-1,-1], [-1,1]])
	trans = TransType(box) 
	x = trans.Transform(trans.Intersect([0.,0], np.array([-2,-1])))
	assert(x==approx([-1.,-.5]))

	box = np.array([[2.,0], [2,1], [0,0], [0,1]])
	trans = TransType(box) 
	x = trans.Transform(trans.Intersect([0.,0], np.array([-1.25,-1])))
	assert(x==approx([3./8,0]))

	theta = np.linspace(0, 2*np.pi, 50)
	for i in range(len(theta)):
		d = np.array([np.cos(theta[i]), np.sin(theta[i])])

	h = .1
	box = np.array([[0,0], [h,0], [0,h], [h,h]])
	trans = TransType(box)
	trans.Intersect([.33998104,-.33998104], np.array([-1,-.25]))

def test_hotrans():
	alpha = .1
	X = np.array([[0.,0], [.5,alpha], [1,0], [0,.5], [.5,.5], [1,.5], [0,1], [.5,1-alpha], [1,1]])
	trans = ElementTrans(X) 
	assert(trans.Transform([0,-1.])==approx([.5,alpha]))
	assert(trans.Transform([1.,1])==approx([1,1]))
	assert(trans.Transform([0,0.])==approx([.5,.5]))
	assert(trans.Transform([-1,0.])==approx([0,.5]))
	assert(trans.Transform([-1,0.])==approx([0,.5]))
	assert(trans.Area()==approx(1-2*alpha+2*alpha/3))