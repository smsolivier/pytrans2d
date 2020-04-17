#!/usr/bin/env python3

import numpy as np

from trans2d import * 
import pytest 

def test_outer():
	v = np.random.rand(4)
	w = np.random.rand(6)

	outer = Outer(1., v,w)
	outer2 = np.outer(v,w)

	assert(np.linalg.norm(outer - outer2)==pytest.approx(0))

def test_outeradd():
	v = np.random.rand(4)
	w = np.random.rand(6)
	z = np.random.random((4,6))

	ex = np.outer(v,w)*2 + z 
	AddOuter(2., v, w, z)

	assert(np.linalg.norm(ex - z)==pytest.approx(0))

def test_mult():
	m = 3 
	n = 2 
	p = 5 

	A = np.random.random((m,n))
	B = np.random.random((n,p))

	C = Mult(2., A, B)
	Cex = np.dot(A, B)*2 
	assert(C==pytest.approx(Cex))

def test_addmult():
	m = 3 
	n = 2 
	p = 5 

	A = np.random.random((m,n))
	B = np.random.random((n,p))
	C = np.random.random((m,p))
	Cex = 3*C + np.dot(A, B)*2 

	AddMult(2., A, B, 3, C)
	assert(C==pytest.approx(Cex))

def test_transmult():
	m = 3 
	n = 2 
	p = 5 

	A = np.random.random((n,m))
	B = np.random.random((n,p))
	C = np.random.random((m,p))

	ex = np.dot(A.transpose(), B)*2 + C 
	AddTransMult(2., A, B, 1., C)

	assert(np.linalg.norm(ex - C)==pytest.approx(0))