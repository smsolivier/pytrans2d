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

@pytest.mark.parametrize('p', np.arange(1,11).tolist())
def test_1D(p):
	c = np.random.rand(p+1)
	cint = np.polynomial.polynomial.polyint(c)
	ex = np.polyval(cint[::-1], 1) - np.polyval(cint[::-1], -1)
	ip, w = quadrature.Get1D(p)
	f = lambda x: np.polyval(c[::-1], x)

	assert(Integrate(f,ip,w)==pytest.approx(ex))

@pytest.mark.parametrize('p', np.arange(1,11).tolist())
def test_2D(p):
	cx = np.random.rand(p+1)
	cy = np.random.rand(p+1)
	cxint = np.polynomial.polynomial.polyint(cx)
	cyint = np.polynomial.polynomial.polyint(cy)
	ex_x = np.polyval(cxint[::-1], 1) - np.polyval(cxint[::-1], -1)
	ex_y = np.polyval(cyint[::-1], 1) - np.polyval(cyint[::-1], -1)
	ex = ex_x * ex_y 
	f = lambda x: np.polyval(cx[::-1], x[0])*np.polyval(cy[::-1], x[1])
	ip, w = quadrature.Get(p)

	assert(Integrate(f,ip,w)==pytest.approx(ex))	