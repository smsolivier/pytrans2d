import numpy as np 
from ..ext.horner import PolyVal2D
from ..ext.horner import PolyValTP
from ..ext import linalg 
from . import basis 

class Element:
	def __init__(self, btype, p):
		self.basis = btype(p)
		self.Nn = self.basis.N**2 

		X,Y = np.meshgrid(self.basis.ip, self.basis.ip) 
		self.nodes = np.zeros((self.Nn, 2))
		self.nodes[:,0] = X.flatten()
		self.nodes[:,1] = Y.flatten()

	def CalcShape(self, xi):
		return PolyVal2D(self.basis.B, self.basis.B, np.array(xi))

	def CalcVShape(self, xi):
		s = self.CalcShape(xi)
		return np.block([[s, np.zeros(s.shape)], [np.zeros(s.shape), s]])

	def CalcPhysVShape(self, trans, xi):
		return self.CalcVShape(xi) 

	def CalcGradShape(self, xi):
		gshape = np.zeros((2,self.Nn))
		gshape[0] = PolyVal2D(self.basis.dB, self.basis.B, np.array(xi))
		gshape[1] = PolyVal2D(self.basis.B, self.basis.dB, np.array(xi))
		return gshape 

	def CalcPhysGradShape(self, trans, xi):
		gs = self.CalcGradShape(xi)
		return linalg.TransMult(trans.Finv(xi), gs)

	def CalcVGradShape(self, xi):
		gs = self.CalcGradShape(xi)
		return np.block([[gs, np.zeros(gs.shape)], [np.zeros(gs.shape), gs]])

	def CalcVPhysGradShape(self, trans, xi):
		pgs = self.CalcPhysGradShape(trans, xi)
		return np.block([[pgs, np.zeros(pgs.shape)], [np.zeros(pgs.shape), pgs]])

	def Interpolate(self, trans, xi, u):
		if (len(u)==self.Nn):
			return np.dot(self.CalcShape(xi), u)
		else:
			return np.dot(self.CalcVShape(xi), u) 

	def InterpolateGradient(self, trans, xi, u):
		pgs = self.CalcPhysGradShape(trans, xi)
		return np.dot(pgs, u) 

class RTElement:
	def __init__(self, bc, bd, p):
		self.p = p 
		self.bx = [bc(p+1), bd(p)]
		self.by = [bd(p), bc(p+1)]

		self.Nn = 2*(p+1)*(p+2) 
		self.nodes = np.zeros((self.Nn, 2)) 

		for i in range(p+1):
			for j in range(p+2):
				idx = j + i*(p+2)
				self.nodes[idx,0] = self.bx[0].ip[j] 
				self.nodes[idx,1] = self.bx[1].ip[i]

		for i in range(p+2):
			for j in range(p+1):
				idx = j + i*(p+1) + (p+1)*(p+2) 
				self.nodes[idx,0] = self.by[0].ip[j] 
				self.nodes[idx,1] = self.by[1].ip[i] 

		self.basis = basis.RTBasis(p) 

	def CalcShape(self, xi):
		raise NotImplementedError('this is vector FE')

	def CalcVShape(self, xi):
		sx = PolyValTP(self.basis.Cx, np.array(xi))
		sy = PolyValTP(self.basis.Cy, np.array(xi))
		# sx = PolyVal2D(self.bx[0].B, self.bx[1].B, np.array(xi))
		# sy = PolyVal2D(self.by[0].B, self.by[1].B, np.array(xi))
		return np.block([[sx, np.zeros(len(sx))], 
			[np.zeros(len(sy)), sy]])

	def CalcPhysVShape(self, trans, xi):
		vs = self.CalcVShape(xi) 
		return 1/trans.Jacobian(xi)*np.dot(trans.F(xi), vs) 

	def CalcDivShape(self, xi):
		dsx = PolyValTP(self.basis.dCx, np.array(xi))
		dsy = PolyValTP(self.basis.dCy, np.array(xi))
		# dsx = PolyVal2D(self.bx[0].dB, self.bx[1].B, np.array(xi))
		# dsy = PolyVal2D(self.by[0].B, self.by[1].dB, np.array(xi))
		return np.concatenate((dsx, dsy)) 

	def CalcPhysDivShape(self, trans, xi):
		ds = self.CalcDivShape(xi)
		return 1/trans.Jacobian(xi)*ds 

	def Interpolate(self, trans, xi, u):
		return np.dot(self.CalcPhysVShape(trans, xi), u) 