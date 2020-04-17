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

def UpwindTraceIntegrator(el1, el2, face, c, qorder):
	elmat = np.zeros((2*el1.Nn, 2*el2.Nn))
	ip, w = quadrature.Get1D(qorder)

	j = np.zeros(el1.Nn+el2.Nn)
	a = np.zeros(el1.Nn+el2.Nn)

	for n in range(len(w)):
		xi1 = face.ipt1.Transform(ip[n]) 
		xi2 = face.ipt2.Transform(ip[n]) 

		s1 = el1.CalcShape(xi1)
		s2 = el2.CalcShape(xi2)
		j[:el1.Nn] = s1 
		j[el1.Nn:] = -s2 

		a[:el1.Nn] = s1 
		a[el1.Nn:] = s2 

		cdotn = np.dot(c, face.face.Normal(ip[n]))
		alpha = .5*face.face.Jacobian(ip[n])*w[n]

		linalg.AddOuter(alpha*cdotn, j, a, elmat)
		linalg.AddOuter(alpha*abs(cdotn), j, j, elmat) 

	return elmat 

def InteriorPenaltyIntegrator(el1, el2, face, c, qorder):
	elmat = np.zeros((2*el1.Nn, 2*el2.Nn))
	ip, w = quadrature.Get1D(qorder)

	j = np.zeros(el1.Nn+el2.Nn)
	a = np.zeros(el1.Nn+el2.Nn)
	for n in range(len(w)):
		xi1 = face.ipt1.Transform(ip[n])
		xi2 = face.ipt2.Transform(ip[n])
		nor = face.face.Normal(ip[n])

		s1 = el1.CalcShape(xi1)
		s2 = el2.CalcShape(xi2)
		pgs1 = el1.CalcPhysGradShape(face.trans1, xi1)
		pgs2 = el2.CalcPhysGradShape(face.trans2, xi2)
		j[:el1.Nn] = s1 
		j[el1.Nn:] = -s2 
		a[:el1.Nn] = np.dot(nor, pgs1)
		a[el1.Nn:] = np.dot(nor, pgs2)

		jac = face.face.Jacobian(ip[n])
		alpha = jac*w[n] 
		linalg.AddOuter(c*alpha/jac**2, j, j, elmat)
		M = linalg.Outer(.5*alpha, j, a)
		sym = .5*(M + M.transpose())
		elmat -= sym 

	return elmat 

def Assemble(space, integrator, c, qorder):
	A = COOMatrix(space.Nu)

	for e in range(space.Ne):
		trans = space.mesh.trans[e] 
		elmat = integrator(space.el, trans, c, qorder)
		A[space.dofs[e], space.dofs[e]] = elmat 

	return A.Get()

def FaceAssemble(space, integrator, c, qorder):
	A = COOMatrix(space.Nu)
	for f in range(len(space.mesh.iface)):
		face = space.mesh.iface[f] 
		elmat = integrator(space.el, space.el, face, c, qorder)
		dofs1 = space.dofs[face.ElNo1] 
		dofs2 = space.dofs[face.ElNo2]
		dofs = np.concatenate((dofs1, dofs2))
		A[dofs, dofs] = elmat 

	return A.Get()

def BdrFaceAssemble(space, integrator, c, qorder):
	A = COOMatrix(space.Nu)
	for f in range(len(space.mesh.bface)):
		face = space.mesh.bface[f]
		elmat = integrator(space.el, space.el, face, c, qorder)
		dofs = space.dofs[face.ElNo1] 
		A[dofs, dofs] = elmat 

	return A.Get()

def FaceAssembleAll(space, integrator, c, qorder):
	return FaceAssemble(space, integrator, c, qorder) + BdrFaceAssemble(space, integrator, c, qorder)
