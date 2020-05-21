#!/usr/bin/env python3

from trans2d import * 
from pytest import approx 
import pytest

@pytest.mark.parametrize('p', [1, 2, 3, 4])
@pytest.mark.parametrize('btype', [LagrangeBasis, LobattoBasis, LegendreBasis])
def test_nodal(btype, p):
	el = Element(btype, p)
	for i in range(el.Nn):
		b = np.zeros(el.Nn)
		b[i] = 1
		assert(el.CalcShape(el.nodes[i,:])==approx(b))

@pytest.mark.parametrize('p1', [1,2,3,4])
@pytest.mark.parametrize('p2', [1,2,3,4])
def test_nodal_mix(p1, p2):
	b1 = LobattoBasis(p1)
	b2 = LegendreBasis(p2)
	N = b1.N*b2.N 
	for i in range(b2.N):
		for j in range(b1.N):
			idx = j + i*b1.N 
			b = np.zeros(N)
			b[idx] = 1 
			s = PolyVal2D(b1.B, b2.B, np.array([b1.ip[j], b2.ip[i]]))
			assert(s==approx(b))

def test_gshape():
	el = Element(LagrangeBasis, 1)
	gs = .25*np.array([[-1,1,-1,1], [-1,-1,1,1]])
	assert(el.CalcGradShape([0.,0])==approx(gs))

@pytest.mark.parametrize('p', [1, 2, 3, 4])
@pytest.mark.parametrize('btype', [LagrangeBasis, LobattoBasis, LegendreBasis])
def test_interpolate(btype, p):
	el = Element(btype, p)
	trans = AffineTrans(np.array([[0,0], [1,0], [1,1], [0,1]]))
	f = lambda x: 10*x[0] + 5*x[1] + x[0]*x[1] 
	df = lambda x: np.array([10 + x[1], 5 + x[0]])

	u = np.zeros(el.Nn)
	for i in range(el.Nn):
		X = trans.Transform(el.nodes[i])
		u[i] = f(X) 

	xi = [.25, .35]
	X = trans.Transform(xi)
	assert(el.Interpolate(xi, u)==approx(f(X)))
	assert(el.InterpolateGradient(trans, xi, u)==approx(df(X)))

@pytest.mark.parametrize('p', [1, 2, 3, 4])
@pytest.mark.parametrize('btype', [LagrangeBasis, LobattoBasis, LegendreBasis])
def test_vinterpolate(btype, p):
	el = Element(btype, p)
	trans = AffineTrans(np.array([[0,0], [1,0], [1,1], [0,1]]))
	f = lambda x: [10*x[0] + x[1], 2*x[0] - x[1]]

	u = np.zeros(2*el.Nn)
	for i in range(el.Nn):
		X = trans.Transform(el.nodes[i])
		evl = f(X)
		u[i] = evl[0]
		u[i+el.Nn] = evl[1] 

	xi = [.25, .35] 
	X = trans.Transform(xi)
	assert(el.Interpolate(xi, u)==approx(np.array(f(X))))
