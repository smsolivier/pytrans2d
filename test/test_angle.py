#!/usr/bin/env python3

from trans2d import * 
import pytest 

def Integrate(f, quad):
	s = 0
	for a in range(quad.N):
		s += quad.w[a] * f(quad.Omega[a])

	return s

@pytest.mark.parametrize('N', [2,4,6,8,10,12,16])
def test_iso(N):
	quad = LevelSym(N)
	f = lambda Omega: 1 

	assert(Integrate(f,quad)==pytest.approx(4*np.pi))

@pytest.mark.parametrize('N', [2,4,6,8,10,12,16])
def test_linear(N):
	quad = LevelSym(N)
	f = lambda Omega: Omega[0] + Omega[1] 

	assert(Integrate(f,quad)==pytest.approx(0))

@pytest.mark.parametrize('N', [2,4,6,8,10,12,16])
def test_quadratic(N):
	quad = LevelSym(N)
	f = lambda Omega: 2*Omega[0]**2 + Omega[1]**2 + Omega[0]*Omega[1] 

	assert(Integrate(f,quad)==pytest.approx(4*np.pi))

