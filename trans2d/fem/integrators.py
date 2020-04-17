import numpy as np 
import scipy.sparse as sp 
import scipy.sparse.linalg as spla 

from . import fespace as fes 
from .quadrature import quadrature 
from ..ext import linalg 

class COOMatrix:
	def __init__(self, m, n=None, zero_tol=1e-14):
		self.m = m 
		self.n = n 
		if (n==None):
			self.n = m 
		self.zero_tol = zero_tol

		self.row = [] 
		self.col = [] 
		self.data = []

	def __setitem__(self, key, item):
		for i in range(len(key[0])):
			for j in range(len(key[1])):
				if (abs(item[i,j])>self.zero_tol):
					self.row.append(key[0][i])
					self.col.append(key[1][j])
					self.data.append(item[i,j])

	def Get(self):
		return sp.coo_matrix((self.data, (self.row, self.col)), (self.m, self.n)).tocsc()

def DiffusionIntegrator(el, trans, c, qorder):
	elmat = np.zeros((el.Nn, el.Nn))
	ip, w = quadrature.Get(qorder)

	for n in range(len(w)):
		pgs = el.CalcPhysGradShape(trans, ip[n]) 
		X = trans.Transform(ip[n])
		linalg.AddTransMult(trans.Jacobian(ip[n])*c(X)*w[n], pgs, pgs, 1., elmat)

	return elmat 

def MassIntegrator(el, trans, c, qorder):
	elmat = np.zeros((el.Nn, el.Nn))
	ip, w = quadrature.Get(qorder)

	for n in range(len(w)):
		s = el.CalcShape(ip[n])
		X = trans.Transform(ip[n]) 
		linalg.AddOuter(trans.Jacobian(ip[n])*c(X)*w[n], s, s, elmat)

	return elmat 

def WeakConvectionIntegrator(el, trans, c, qorder):
	elmat = np.zeros((el.Nn, el.Nn))
	ip, w = quadrature.Get(qorder)

	for n in range(len(w)):
		pgs = el.CalcPhysGradShape(trans, ip[n]) 
		s = el.CalcShape(ip[n])
		cpgs = np.dot(c, pgs)
		linalg.AddOuter(-trans.Jacobian(ip[n])*w[n], cpgs, s, elmat) 

	return elmat 

def DomainIntegrator(el, trans, c, qorder):
	elvec = np.zeros(el.Nn)
	ip, w = quadrature.Get(qorder)

	for n in range(len(w)):
		s = el.CalcShape(ip[n])
		X = trans.Transform(ip[n])
		elvec += s * c(X) * trans.Jacobian(ip[n]) * w[n] 

	return elvec 


def Assemble(space, integrator, c, qorder):
	A = COOMatrix(space.Nu)

	for e in range(space.Ne):
		trans = space.mesh.trans[e] 
		elmat = integrator(space.el, trans, c, qorder)
		A[space.dofs[e], space.dofs[e]] = elmat 

	return A.Get()

def AssembleRHS(space, integrator, c, qorder):
	v = np.zeros(space.Nu)

	for e in range(space.Ne):
		trans = space.mesh.trans[e]
		elvec = integrator(space.el, trans, c, qorder)
		v[space.dofs[e]] += elvec 

	return v 