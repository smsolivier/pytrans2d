import numpy as np 
import scipy.sparse as sp 
import scipy.sparse.linalg as spla 

from . import fespace as fes 
from .quadrature import quadrature 
from ..ext import linalg 
from .element import Element

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

class BlockMatrix:
	def __init__(self, lst):
		self.lst = lst 

	def __iadd__(self, other):
		for i in range(len(self.lst)):
			for j in range(len(self.lst[i])):
				self.lst[i][j] += other.lst[i][j] 

		return self

	def __add__(self, a):
		lst = [[] for i in self.lst]
		for i in range(len(self.lst)):
			for j in range(len(self.lst[i])):
				A = self[i,j] 
				B = a[i,j]
				lst[i].append(A+B)

		return BlockMatrix(lst)

	def __getitem__(self, key):
		return self.lst[key[1]][key[0]]

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

def InverseMassIntegrator(el, trans, c, qorder):
	M = MassIntegrator(el, trans, c, qorder)
	return np.linalg.inv(M)

def MassIntegratorRowSum(el, trans, c, qorder):
	M = MassIntegrator(el, trans, c, qorder)
	for i in range(M.shape[0]):
		tmp = 0 
		for j in range(M.shape[1]):
			tmp += M[i,j]
			M[i,j] = 0
		M[i,i] = tmp 

	return M 

def MixMassIntegrator(el1, el2, trans, c, qorder):
	elmat = np.zeros((el1.Nn, el2.Nn))
	ip, w = quadrature.Get(qorder)

	for n in range(len(w)):
		s1 = el1.CalcShape(ip[n])
		s2 = el2.CalcShape(ip[n])
		X = trans.Transform(ip[n])
		linalg.AddOuter(trans.Jacobian(ip[n])*c(X)*w[n], s1, s2, elmat) 

	return elmat 

def VectorMassIntegrator(el, trans, c, qorder):
	elmat = np.zeros((2*el.Nn, 2*el.Nn))
	ip, w = quadrature.Get(qorder)
	M = MassIntegrator(el, trans, c, qorder)
	elmat[:el.Nn,:el.Nn] = M
	elmat[el.Nn:,el.Nn:] = M 
	return elmat 

def InverseVectorMassIntegrator(el, trans, c, qorder):
	elmat = np.zeros((2*el.Nn, 2*el.Nn))
	ip, w = quadrature.Get(qorder)
	M = MassIntegrator(el, trans, c, qorder)
	Minv = np.linalg.inv(M) 
	elmat[:el.Nn,:el.Nn] = Minv
	elmat[el.Nn:,el.Nn:] = Minv 
	return elmat 	

def VectorFEMassIntegrator(el, trans, c, qorder):
	ip, w = quadrature.Get(qorder)
	elmat = np.zeros((el.Nn, el.Nn))
	for n in range(len(w)):
		vs = el.CalcVShape(ip[n])
		F = trans.F(ip[n]) 
		G = linalg.TransMult(F, F) 
		vshat = np.dot(G, vs)
		X = trans.Transform(ip[n])
		linalg.AddTransMult(w[n]/trans.Jacobian(ip[n])*c(X), 
			vs, vshat, 1., elmat)

	return elmat 

def VectorFEMassIntegratorRowSum(el, trans, c, qorder):
	M = VectorFEMassIntegrator(el, trans, c, qorder)
	for i in range(M.shape[0]):
		tmp = 0 
		for j in range(M.shape[1]):
			tmp += M[i,j]
			M[i,j] = 0
		M[i,i] = tmp 

	return M 	

def VectorMassIntegratorRowSum(el, trans, c, qorder):
	M = VectorMassIntegrator(el, trans, c, qorder)
	for i in range(M.shape[0]):
		tmp = 0
		for j in range(M.shape[1]):
			tmp += M[i,j] 
			M[i,j] = 0 
		M[i,i] = tmp 

	return M 

def WeakConvectionIntegrator(el, trans, c, qorder):
	elmat = np.zeros((el.Nn, el.Nn))
	ip, w = quadrature.Get(qorder)

	for n in range(len(w)):
		pgs = el.CalcPhysGradShape(trans, ip[n]) 
		s = el.CalcShape(ip[n])
		cpgs = np.dot(c, pgs)
		linalg.AddOuter(-trans.Jacobian(ip[n])*w[n], cpgs, s, elmat) 

	return elmat 

def MixDivIntegrator(el1, el2, trans, c, qorder):
	elmat = np.zeros((el1.Nn, 2*el2.Nn))
	ip, w = quadrature.Get(qorder)

	for n in range(len(w)):
		s = el1.CalcShape(ip[n])
		div = el2.CalcPhysGradShape(trans, ip[n]).flatten()
		linalg.AddOuter(trans.Jacobian(ip[n])*w[n]*c, s, div, elmat)

	return elmat 

def DivDivIntegrator(el, trans, c, qorder):
	elmat = np.zeros((el.Nn*2, el.Nn*2))
	ip, w = quadrature.Get(qorder)

	for n in range(len(w)):
		div = el.CalcPhysGradShape(trans, ip[n]).flatten()
		linalg.AddOuter(trans.Jacobian(ip[n])*w[n]*c, div, div, elmat)

	return elmat 

def VectorFEDivIntegrator(el1, el2, trans, c, qorder):
	elmat = np.zeros((el1.Nn, el2.Nn))
	ip, w = quadrature.Get(qorder)

	for n in range(len(w)):
		s = el1.CalcShape(ip[n])
		div = el2.CalcDivShape(ip[n])
		linalg.AddOuter(w[n]*c, s, div, elmat) 

	return elmat 

def VectorFEDivIntegrator2(el1, el2, trans, c, qorder):
	elmat = np.zeros((el1.Nn, el2.Nn))
	ip, w = quadrature.Get(qorder)

	for n in range(len(w)):
		gs = el1.CalcVGradShape(trans, ip[n]) 
		D = np.linalg.multi_dot([trans.F(ip[n]).T, c(trans.Transform(ip[n])), trans.Finv(ip[n]).T])
		# D = c(trans.Transform(ip[n]))
		gsd = gs.T@D.flatten()
		s = el2.CalcShape(ip[n]) 
		linalg.AddOuter(-w[n], gsd, s, elmat) 

	return elmat 

def WeakMixDivIntegrator(el1, el2, trans, c, qorder):
	elmat = np.zeros((el1.Nn, 2*el2.Nn))
	ip, w = quadrature.Get(qorder)

	for n in range(len(w)):
		pgs = el1.CalcPhysGradShape(trans, ip[n])
		vs = el2.CalcVShape(ip[n])
		linalg.AddTransMult(-trans.Jacobian(ip[n])*w[n]*c, pgs, vs, 1., elmat)

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
		M = linalg.Outer((1 if face.boundary else .5)*alpha, j, a)
		sym = M + M.transpose()
		elmat -= sym 

	return elmat 

def PenaltyIntegrator(el1, el2, face, c, qorder):
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
		j[:el1.Nn] = s1 
		j[el1.Nn:] = -s2 

		jac = face.face.Jacobian(ip[n])
		alpha = jac*w[n] 
		linalg.AddOuter(c*alpha/jac**2, j, j, elmat)

	return elmat 	

def Assemble(space, integrator, c, qorder):
	A = COOMatrix(space.Nu)

	for e in range(space.Ne):
		trans = space.mesh.trans[e] 
		elmat = integrator(space.el, trans, c, qorder)
		A[space.dofs[e], space.dofs[e]] = elmat 

	return A.Get()

def AssembleBlocks(space, integrator, c, qorder):
	nu = int(space.Nu/2)
	A = COOMatrix(nu)
	B = COOMatrix(nu)
	C = COOMatrix(nu)
	D = COOMatrix(nu) 
	for e in range(space.Ne):
		trans = space.mesh.trans[e]
		elmat = integrator(space.el, trans, c, qorder)
		half = int(elmat.shape[0]/2)
		dofs = space.dofs[e][:half]
		A[dofs,dofs] = elmat[:half, :half]
		B[dofs,dofs] = elmat[:half, half:]
		C[dofs,dofs] = elmat[half:, :half]
		D[dofs,dofs] = elmat[half:, half:]

	return BlockMatrix([[A.Get(), B.Get()], [C.Get(), D.Get()]])

def AssembleLOR(space, integrator, c, qorder):
	# lomesh = space.LORefine().mesh 
	el = Element(space.btype, 1)
	p = space.el.basis.p 
	A = COOMatrix(space.Nu)

	for e in range(space.Ne):
		trans = space.mesh.trans[e]
		for i in range(p):
			for j in range(p):
				nid = np.array([i*(p+1)+j, i*(p+1)+j+1, (i+1)*(p+1)+j, (i+1)*(p+1)+j+1])
				box = np.zeros((len(nid), 2))
				for n in range(len(nid)):
					box[n] = trans.Transform(space.el.nodes[nid[n]])

				rtrans = type(trans)(box) 
				elmat = integrator(el, rtrans, c, qorder)
				dofid = nid.copy()
				if (space.vdim>1):
					dofid = np.append(dofid, dofid+(p+1)**2) 
				dofs = space.dofs[e,dofid] 
				A[dofs, dofs] = elmat 

	return A.Get()

def FaceAssemble(space, integrator, c, qorder):
	A = COOMatrix(space.Nu)
	for face in space.mesh.iface:
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

def BdrFaceAssembleBlocks(space, integrator, c, qorder):
	nu = int(space.Nu/2)
	A = COOMatrix(nu)
	B = COOMatrix(nu)
	C = COOMatrix(nu)
	D = COOMatrix(nu) 
	for face in space.mesh.bface:
		elmat = integrator(space.el, space.el, face, c, qorder)
		half = int(elmat.shape[0]/2)
		dofs = space.dofs[face.ElNo1][:half]
		A[dofs,dofs] = elmat[:half, :half]
		B[dofs,dofs] = elmat[:half, half:]
		C[dofs,dofs] = elmat[half:, :half]
		D[dofs,dofs] = elmat[half:, half:]

	return BlockMatrix([[A.Get(), B.Get()], [C.Get(), D.Get()]])

def FaceAssembleAll(space, integrator, c, qorder):
	return FaceAssemble(space, integrator, c, qorder) + BdrFaceAssemble(space, integrator, c, qorder)

def MixFaceAssemble(space1, space2, integrator, c, qorder):
	A = COOMatrix(space1.Nu, space2.Nu)
	for face in space1.mesh.iface:
		elmat = integrator([space1.el, space1.el], [space2.el, space2.el], face, c, qorder)
		dofs11 = space1.dofs[face.ElNo1]
		dofs12 = space1.dofs[face.ElNo2] 
		dofs21 = space2.dofs[face.ElNo1]
		dofs22 = space2.dofs[face.ElNo2]
		dofs1 = np.concatenate((dofs11, dofs12))
		dofs2 = np.concatenate((dofs21, dofs22))
		assert((len(dofs1), len(dofs2))==elmat.shape)
		A[dofs1, dofs2] = elmat 

	return A.Get()

def BdrMixFaceAssemble(space1, space2, integrator, c, qorder):
	A = COOMatrix(space1.Nu, space2.Nu)
	for face in space1.mesh.bface:
		elmat = integrator([space1.el, space1.el], [space2.el, space2.el], face, c, qorder)
		dofs11 = space1.dofs[face.ElNo1]
		dofs21 = space2.dofs[face.ElNo1]
		A[dofs11, dofs21] = elmat 

	return A.Get()

def MixFaceAssembleAll(space1, space2, integrator, c, qorder):
	return MixFaceAssemble(space1, space2, integrator, c, qorder) \
		+ BdrMixFaceAssemble(space1, space2, integrator, c, qorder)

def MixAssemble(space1, space2, integrator, c, qorder):
	A = COOMatrix(space1.Nu, space2.Nu)
	for e in range(space1.Ne):
		trans = space1.mesh.trans[e]
		elmat = integrator(space1.el, space2.el, trans, c, qorder)
		A[space1.dofs[e], space2.dofs[e]] = elmat 

	return A.Get()