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
	assert(el.Interpolate(trans, xi, u)==approx(f(X)))
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
	assert(el.Interpolate(trans, xi, u)==approx(np.array(f(X))))

@pytest.mark.parametrize('p', [0,1,2,3])
def test_rtmodal(p):
	qorder = 2*(p+1)+1
	ip, w = quadrature.Get1D(qorder)
	el = RTElement(LobattoBasis, LegendreBasis, p)

	# bottom face zeroth moment 
	ii = np.zeros(el.Nn)
	for n in range(len(w)):
		vs = el.CalcVShape([ip[n], -1.]) 
		nor = np.array([0,1.])
		s = np.dot(nor, vs) 
		ii += s * w[n] 

	s = np.zeros(el.Nn)
	s[int(el.Nn/2)] = 1 
	assert(ii==approx(s))

	# bottom face first moment 
	if (p>0):
		ii = np.zeros(el.Nn)
		for n in range(len(w)):
			vs = el.CalcVShape([ip[n], -1.]) 
			nor = np.array([0,1])
			s = np.dot(nor, vs) 
			ii += s * w[n] * ip[n]

		s = np.zeros(el.Nn)
		s[int(el.Nn/2)+1] = 1 
		assert(ii==approx(s))

	# bottom face second moment 
	if (p>1):
		ii = np.zeros(el.Nn)
		for n in range(len(w)):
			vs = el.CalcVShape([ip[n],-1]) 
			nor = np.array([0,1])
			s = np.dot(nor, vs) 
			ii += s * w[n] * ip[n]**2

		s = np.zeros(el.Nn)
		s[int(el.Nn/2)+2] = 1 
		assert(ii==approx(s))

	# top face zeroth moment 
	ii = np.zeros(el.Nn)
	for n in range(len(w)):
		vs = el.CalcVShape([ip[n], 1.]) 
		nor = np.array([0,1.])
		s = np.dot(nor, vs) 
		ii += s * w[n] 

	s = np.zeros(el.Nn)
	s[int(el.Nn/2)+(p+1)**2] = 1 
	assert(ii==approx(s))

	# top face first moment 
	if (p>0):
		ii = np.zeros(el.Nn)
		for n in range(len(w)):
			vs = el.CalcVShape([ip[n], 1.]) 
			nor = np.array([0,1])
			s = np.dot(nor, vs) 
			ii += s * w[n] * ip[n]

		s = np.zeros(el.Nn)
		s[int(el.Nn/2)+(p+1)**2+1] = 1 
		assert(ii==approx(s))

	# top face second moment 
	if (p>1):
		ii = np.zeros(el.Nn)
		for n in range(len(w)):
			vs = el.CalcVShape([ip[n],1]) 
			nor = np.array([0,1])
			s = np.dot(nor, vs) 
			ii += s * w[n] * ip[n]**2

		s = np.zeros(el.Nn)
		s[int(el.Nn/2)+(p+1)**2+2] = 1 
		assert(ii==approx(s))

	# left face zeroth moment 
	ii = np.zeros(el.Nn)
	for n in range(len(w)):
		vs = el.CalcVShape([-1., ip[n]]) 
		nor = np.array([1,0])
		s = np.dot(nor, vs) 
		ii += s * w[n] 

	s = np.zeros(el.Nn)
	s[0] = 1 
	assert(ii==approx(s))

	# left face first moment 
	if (p>0):
		ii = np.zeros(el.Nn)
		for n in range(len(w)):
			vs = el.CalcVShape([-1., ip[n]]) 
			nor = np.array([1,0])
			s = np.dot(nor, vs) 
			ii += s * w[n] * ip[n]

		s = np.zeros(el.Nn)
		s[p+2] = 1 
		assert(ii==approx(s))

	# left face second moment 
	if (p>1):
		ii = np.zeros(el.Nn)
		for n in range(len(w)):
			vs = el.CalcVShape([-1., ip[n]]) 
			nor = np.array([1,0])
			s = np.dot(nor, vs) 
			ii += s * w[n] * ip[n]**2

		s = np.zeros(el.Nn)
		s[2*(p+2)] = 1 
		assert(ii==approx(s))

	# right face zeroth moment 
	ii = np.zeros(el.Nn)
	for n in range(len(w)):
		vs = el.CalcVShape([1., ip[n]]) 
		nor = np.array([1,0])
		s = np.dot(nor, vs) 
		ii += s * w[n] 

	s = np.zeros(el.Nn)
	s[p+1] = 1 
	assert(ii==approx(s))

	# right face first moment 
	if (p>0):
		ii = np.zeros(el.Nn)
		for n in range(len(w)):
			vs = el.CalcVShape([1., ip[n]]) 
			nor = np.array([1,0])
			s = np.dot(nor, vs) 
			ii += s * w[n] * ip[n]

		s = np.zeros(el.Nn)
		s[p+1+p+2] = 1 
		assert(ii==approx(s))

	# right face second moment 
	if (p>1):
		ii = np.zeros(el.Nn)
		for n in range(len(w)):
			vs = el.CalcVShape([1., ip[n]]) 
			nor = np.array([1,0])
			s = np.dot(nor, vs) 
			ii += s * w[n] * ip[n]**2

		s = np.zeros(el.Nn)
		s[p+1+2*(p+2)] = 1 
		assert(ii==approx(s))

	if (p>0):
		ip, w = quadrature.Get(qorder)
		ix = np.zeros(el.Nn)
		iy = np.zeros(el.Nn)
		for n in range(len(w)):
			vs = el.CalcVShape(ip[n]) 
			sx = vs[0] 
			sy = vs[1]
			ix += sx * w[n] 
			iy += sy * w[n] 

		s = np.zeros(el.Nn)
		s[1] = 1
		assert(ix==approx(s))
		s = np.zeros(el.Nn)
		s[int(el.Nn/2)+p+1] = 1
		assert(iy==approx(s))

def test_rtdiv():
	p = 0 
	el = RTElement(LobattoBasis, LegendreBasis, p)
	div = np.array([-1,1,-1,1])/4
	assert(el.CalcDivShape([0,0.])==approx(div))

def test_rtinterp():
	p = 0 
	el = RTElement(LobattoBasis, LegendreBasis, p) 
	dof = np.array([-4,8,4,-4])
	vs = el.CalcVShape([-1,1])
	val = np.dot(vs, dof)
	assert(val==pytest.approx(np.array([-2,-2])))