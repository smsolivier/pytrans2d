import numpy as np 
from ..ext.horner import PolyVal2D
from ..ext.horner import PolyValTP
from ..ext import linalg 
from . import basis 

class Element:
	def __init__(self, btype, p):
		self.basis = btype(p)
		self.Nn = self.basis.N**2 
		self.p = p 

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

	def Plot(self, trans, u, N=30):
		xi1d = np.linspace(-1,1,N)
		Xi,Eta = np.meshgrid(xi1d,xi1d)
		X = np.zeros(Xi.shape)
		Y = np.zeros(Eta.shape)
		U = np.zeros(Xi.shape)
		for i in range(N):
			for j in range(N):
				xi = np.array([Xi[i,j], Eta[i,j]])
				U[i,j] = self.Interpolate(trans, xi, u)
				X[i,j], Y[i,j] = trans.Transform(xi)

		return X,Y,U 

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
		self.modal = False

	def CalcShape(self, xi):
		raise NotImplementedError('this is vector FE')

	def CalcVShape(self, xi):
		if (self.modal):
			sx = PolyValTP(self.basis.Cx, np.array(xi))
			sy = PolyValTP(self.basis.Cy, np.array(xi))
		else:
			sx = PolyVal2D(self.bx[0].B, self.bx[1].B, np.array(xi))
			sy = PolyVal2D(self.by[0].B, self.by[1].B, np.array(xi))
		return np.block([[sx, np.zeros(len(sx))], 
			[np.zeros(len(sy)), sy]])

	def CalcPhysVShape(self, trans, xi):
		vs = self.CalcVShape(xi) 
		return 1/trans.Jacobian(xi)*np.dot(trans.F(xi), vs) 

	def CalcVGradShape(self, trans, xi):
		gsx = np.zeros((2,int(self.Nn/2)))
		gsy = np.zeros((2,int(self.Nn/2)))
		H = trans.H(xi)
		T1 = np.array([[H[1,1], H[1,2]], [-H[1,0], -H[1,1]]])
		T2 = np.array([[-H[0,1], -H[0,2]], [H[0,0], H[0,1]]])
		T = np.zeros((4,2))
		T[:,0] = T1.flatten()
		T[:,1] = T2.flatten()
		if (self.modal):
			gsx[0] = PolyValTP(self.basis.dCx, np.array(xi))
			gsx[1] = PolyValTP(self.basis.dCx2, np.array(xi))
			gsy[0] = PolyValTP(self.basis.dCy2, np.array(xi))
			gsy[1] = PolyValTP(self.basis.dCy, np.array(xi))
		else:
			gsx[0] = PolyVal2D(self.bx[0].dB, self.bx[1].B, np.array(xi))
			gsx[1] = PolyVal2D(self.bx[0].B, self.bx[1].dB, np.array(xi))
			gsy[0] = PolyVal2D(self.by[0].dB, self.by[1].B, np.array(xi))
			gsy[1] = PolyVal2D(self.by[0].B, self.by[1].dB, np.array(xi))
		vs = self.CalcPhysVShape(trans, xi) 
		return -T@vs + np.block([[gsx, np.zeros(gsx.shape)], [np.zeros(gsy.shape), gsy]])

	def CalcDivShape(self, xi):
		if (self.modal):
			dsx = PolyValTP(self.basis.dCx, np.array(xi))
			dsy = PolyValTP(self.basis.dCy, np.array(xi))
		else:
			dsx = PolyVal2D(self.bx[0].dB, self.bx[1].B, np.array(xi))
			dsy = PolyVal2D(self.by[0].B, self.by[1].dB, np.array(xi))
		return np.concatenate((dsx, dsy)) 

	def CalcPhysDivShape(self, trans, xi):
		ds = self.CalcDivShape(xi)
		return 1/trans.Jacobian(xi)*ds 

	def Interpolate(self, trans, xi, u):
		return np.dot(self.CalcPhysVShape(trans, xi), u) 