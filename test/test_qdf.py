#!/usr/bin/env python3

from trans2d import * 
import pytest 

mesh = RectMesh(3,3, [3,3])
space = L2Space(mesh, LagrangeBasis, 1)
quad = LevelSym(16)
psi = TVector(space, quad) 
qdf = QDFactors(space, quad)

def test_isotropic():
	psi.Project(lambda x, Omega: 1)
	qdf.Compute(psi) 

	trans = mesh.trans[0]
	E = qdf.EvalTensor(trans, [0,0])
	assert(E==pytest.approx(np.eye(2)/3))

	fi = mesh.iface[0]
	G = qdf.EvalG(fi, 0)
	assert(G==pytest.approx(.5, abs=1e-2))

def test_quadratic():
	mesh = RectMesh(3,3, [1,1])
	space = L2Space(mesh, LagrangeBasis, 5)
	psi = TVector(space, quad) 
	alpha = lambda x: np.sin(np.pi*x[0])*np.sin(np.pi*x[1]) + 10
	beta = lambda x: np.sin(np.pi*(x[0]+.1)/1.2)*np.sin(np.pi*(x[1]+.1)/1.2)
	gamma = lambda x: x[0]*(1-x[0])*x[1]*(1-x[1])
	# delta = lambda x: np.sin(2*np.pi*(x[0]+.2)/1.4)*np.sin(2*np.pi*(x[1]+.2)/1.4)
	delta = lambda x: 2*x[0] + x[1] + 1
	psi_ex = lambda x, Omega: 1/4/np.pi*(alpha(x) + (Omega[0] + Omega[1])*beta(x) 
		+ Omega[0]*Omega[1]*gamma(x) + Omega[0]**2*delta(x))
	qdf = QDFactors(space, quad, psi_ex)
	def E_ex(x):
		P = np.array([[5*alpha(x) + 3*delta(x), gamma(x)], [gamma(x), 5*alpha(x)+delta(x)]])
		b = 15*alpha(x) + 5*delta(x)
		return P/b 
	def G_ex(x,nor):
		return 3/8*(4*alpha(x) + delta(x))/(3*alpha(x) + delta(x))
	Jin_ex = lambda x: 1/48*(-12*alpha(x) - 8*beta(x) - 3*delta(x))

	psi.Project(psi_ex)
	qdf.Compute(psi)
	trans = mesh.trans[0]
	xi = [.25, -.1]
	X = trans.Transform(xi)
	E = qdf.EvalTensor(trans, xi)
	assert(E==pytest.approx(E_ex(X)))

	fi = mesh.bface[0]
	ip = .25 
	xi = fi.ipt1.Transform(ip)
	X = fi.trans1.Transform(xi)
	nor = fi.face.Normal(ip)
	G = qdf.EvalG(fi, ip) 
	assert(G==pytest.approx(G_ex(X, fi.face.Normal(ip)), abs=1e-1))

	Jin = qdf.EvalJinBdr(fi, ip)
	assert(Jin==pytest.approx(Jin_ex(X), rel=1e-2))

def test_weakedddiv():
	psi.Project(lambda x, Omega: 1)
	qdf.Compute(psi)

	el1 = Element(LagrangeBasis, 1)
	el2 = Element(LagrangeBasis, 0) 

	trans = mesh.trans[0]
	G = WeakEddDivIntegrator(el1, el2, trans, qdf, 3)
	D = MixDivIntegrator(el2, el1, trans, 1, 3)
	assert(G==pytest.approx(-1/3*D.transpose()))

def test_negative():
	psi.Project(lambda x, Omega: np.sin(np.pi*x[0])*np.sin(np.pi*x[1]))
	for i in range(3):
		with pytest.warns(NegativityWarning):
			qdf.Compute(psi) 
