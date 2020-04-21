import numpy as np 
from ..ext.horner import PolyVal2D

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
		N = np.zeros((2, 2*self.Nn))
		N[0,:self.Nn] = s 
		N[1,self.Nn:] = s 
		return N

	def CalcGradShape(self, xi):
		gshape = np.zeros((2,self.Nn))
		gshape[0] = PolyVal2D(self.basis.dB, self.basis.B, np.array(xi))
		gshape[1] = PolyVal2D(self.basis.B, self.basis.dB, np.array(xi))
		return gshape 

	def CalcPhysGradShape(self, trans, xi):
		gs = self.CalcGradShape(xi)
		return np.dot(trans.FinvT(xi), gs)

	def CalcVGradShape(self, xi):
		gs = self.CalcGradShape(xi)
		vgs = np.zeros((4, 2*self.Nn))
		vgs[0:2,:self.Nn] = gs 
		vgs[2:4,self.Nn:] = gs 
		return vgs 

	def CalcVPhysGradShape(self, trans, xi):
		pgs = self.CalcPhysGradShape(trans, xi)
		vpgs = np.zeros((4, 2*self.Nn))
		vpgs[0:2,:self.Nn] = pgs 
		vpgs[2:4,self.Nn:] = pgs 
		return vpgs 

	def Interpolate(self, xi, u):
		if (len(u)==self.Nn):
			return np.dot(self.CalcShape(xi), u)
		else:
			return np.dot(self.CalcVShape(xi), u) 

	def InterpolateGradient(self, trans, xi, u):
		pgs = self.CalcPhysGradShape(trans, xi)
		return np.dot(pgs, u) 
