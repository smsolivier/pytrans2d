import numpy as np 

from . import fespace as fes 
from .quadrature import quadrature 
from ..ext import linalg 

def DomainIntegrator(el, trans, c, qorder):
	elvec = np.zeros(el.Nn)
	ip, w = quadrature.Get(qorder)

	for n in range(len(w)):
		s = el.CalcShape(ip[n])
		X = trans.Transform(ip[n])
		elvec += s * c(X) * trans.Jacobian(ip[n]) * w[n] 

	return elvec 

def InflowIntegrator(el, face, c, qorder):
	elvec = np.zeros(el.Nn)
	ip, w = quadrature.Get1D(qorder)
	Omega = c[0]
	inflow = c[1] 

	for n in range(len(w)):
		nor = face.face.Normal(ip[n])
		odotn = np.dot(Omega, nor)
		if (odotn<0):
			xi = face.ipt1.Transform(ip[n])
			X = face.trans1.Transform(xi)
			s = el.CalcShape(xi)
			elvec -= face.face.Jacobian(ip[n])*w[n]*odotn*inflow(X,Omega) * s

	return elvec 

def AssembleRHS(space, integrator, c, qorder):
	v = np.zeros(space.Nu)

	for e in range(space.Ne):
		trans = space.mesh.trans[e]
		elvec = integrator(space.el, trans, c, qorder)
		v[space.dofs[e]] += elvec 

	return v 

def FaceAssembleRHS(space, integrator, c, qorder):
	v = np.zeros(space.Nu)

	for face in space.mesh.bface:
		elvec = integrator(space.el, face, c, qorder)
		v[space.dofs[face.ElNo1]] += elvec 

	return v 