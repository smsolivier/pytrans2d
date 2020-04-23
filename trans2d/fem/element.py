import numpy as np 
from ..ext.horner import PolyVal2D
from ..ext import linalg 

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

	def Interpolate(self, xi, u):
		if (len(u)==self.Nn):
			return np.dot(self.CalcShape(xi), u)
		else:
			return np.dot(self.CalcVShape(xi), u) 

	def InterpolateGradient(self, trans, xi, u):
		pgs = self.CalcPhysGradShape(trans, xi)
		return np.dot(pgs, u) 
