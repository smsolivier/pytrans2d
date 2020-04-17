#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from trans2d import * 
import pytest 

def Integrate(f, ip, w):
	s = 0
	for n in range(len(w)):
		s += f(ip[n]) * w[n] 

	return s	

@pytest.mark.parametrize('p', [1,2,3,4,5])
def test_1D(p):
	c = np.random.rand(p+1)
	cint = np.polynomial.polynomial.polyint(c)
	ex = np.polyval(cint[::-1], 1) - np.polyval(cint[::-1], -1)
	ip, w = quadrature.Get1D(p)
	f = lambda x: np.polyval(c[::-1], x)

	assert(Integrate(f,ip,w)==pytest.approx(ex))

@pytest.mark.parametrize('p', [1,2,3,4,5])
def test_2D(p):
	f = lambda x: 1 + x[0] + 2*x[1] + x[0]*x[1] 
	ip, w = quadrature.Get(p)

	assert(Integrate(f,ip,w)==pytest.approx(4))	