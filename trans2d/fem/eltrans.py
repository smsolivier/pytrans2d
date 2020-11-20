import numpy as np 
import warnings 

from ..ext.horner import PolyVal2D
from ..ext import linalg 
from .basis import LagrangeBasis
from .. import utils

class AffineTrans:
	def __init__(self, box, elno=-1):
		self.box = box 
		self.ElNo = elno 
		x = self.box[:,0]
		y = self.box[:,1]
		self.f = np.array([[-x[0]+x[1]-x[2]+x[3], -x[0]-x[1]+x[2]+x[3]],
			[-y[0]+y[1]-y[2]+y[3], -y[0]-y[1]+y[2]+y[3]]])/4
		self.j = np.linalg.det(self.f)
		self.finv = np.linalg.inv(self.f)
		self.finvT = self.finv.transpose()
		self.c = np.array([np.sum(x), np.sum(y)])/4

	def Transform(self, xi):
		return self.f@xi + self.c

	def Jacobian(self, xi):
		return self.j

	def Area(self):
		return 4*self.j 

	def Length(self):
		return 2*np.sqrt(self.j) 

	def F(self, xi):
		return self.f 

	def Finv(self, xi):
		return self.finv 

	def FinvT(self, xi):
		return self.finvT 

	def InverseMap(self, x, niter=20, tol=1e-14):
		return np.dot(self.finv, x - self.c)

	def Intersect(self, ip, d, niter=1):
		dhat = self.finv@d 
		t = np.zeros(4) - 10
		if (dhat[0]!=0):
			t[1] = (1-ip[0])/dhat[0]		
			t[3] = (-1-ip[0])/dhat[0]
		if (dhat[1]!=0):		
			t[0] = (-1-ip[1])/dhat[1]
			t[2] = (1-ip[1])/dhat[1]
		for i in range(4):
			if (t[i]>=0):
				xi = ip + t[i]*dhat 
				if (xi[0]>=-1. and xi[0]<=1. and xi[1]>=-1. and xi[1]<=1.):
					return xi 

		raise RuntimeError('intersection not found')

class LinearTrans:
	def __init__(self, box, elno=-1):
		self.box = box
		self.ElNo = elno
		self.basis = LagrangeBasis(1)

	def Transform(self, xi):
		shape = PolyVal2D(self.basis.B, self.basis.B, np.array(xi))
		return np.dot(shape, self.box)

	def F(self, xi):
		gs = np.zeros((2, 4))
		gs[0,:] = PolyVal2D(self.basis.dB, self.basis.B, np.array(xi))
		gs[1,:] = PolyVal2D(self.basis.B, self.basis.dB, np.array(xi))
		# hard copy required for linalg functions to avoid 'soft transpose'
		return linalg.Mult(1., gs, self.box).transpose().copy(order='C')

	def Finv(self, xi):
		F = self.F(xi)
		return np.linalg.inv(F) 

	def FinvT(self, xi):
		F = self.F(xi)
		return np.linalg.inv(F).transpose()

	def Jacobian(self, xi):
		return np.linalg.det(self.F(xi))

	def Area(self):
		from .quadrature import quadrature 
		ip, w = quadrature.Get(2)
		area = 0
		for n in range(len(w)):
			area += self.Jacobian(ip[n])*w[n] 

		return area 

	def InverseMap(self, x, niter=20, tol=1e-14):
		xi = np.array([0.,0.])
		for n in range(niter):
			xi0 = xi.copy()
			xi = np.dot(self.Finv(xi0), (x - self.Transform(xi0))) + xi0 
			norm = np.linalg.norm(xi - xi0)
			if (norm < tol):
				break 

		if (norm>tol):
			warnings.warn('inverse map not converged. final tol = {:.3e}'.format(norm), utils.ToleranceWarning)

		return xi 

	def Intersect(self, ip, d, niter=20, tol=1e-14):
		raise NotImplementedError 

		# def RayTrace(trans, ip, Omega):
		# 	found = False 
		# 	IP = trans.Transform(ip) 
		# 	for i in range(2):
		# 		xi = np.zeros(2)
		# 		xi[i] = -1 
		# 		for n in range(25):
		# 			F = trans.F(xi)
		# 			x = trans.Transform(xi) 
		# 			rhs = IP - x + np.dot(F, xi)
		# 			rhs = np.append(rhs, -1) 
		# 			A = np.zeros((3,3))
		# 			A[:2,:2] = F 
		# 			A[:2,-1] = Omega 
		# 			A[-1,i] = 1 

		# 			u = np.linalg.solve(A, rhs) 
		# 			xi = u[:2] 
		# 			t = u[-1] 
		# 			norm = np.linalg.norm(trans.Transform(xi) - (IP - t*Omega))
		# 			if (norm<1e-8):
		# 				break 

		# 		if (xi[(i+1)%2] >= -1. and xi[(i+1)%2] <= 1. and t>=0 and norm<1e-7):
		# 			found = True
		# 			break 

		# 	if not(found):
		# 		raise RuntimeError('intersection not found. ip = ({:.2f},{:.2f}), J = {:.3f}'.format(
		# 			ip[0], ip[1], trans.Jacobian(ip))) 

		# 	return xi 	

class AffineFaceTrans:
	def __init__(self, line):
		self.line = line 
		self.F = np.dot(.5*np.array([-1,1]), line)
		self.c = .5*np.array([line[1,0] + line[0,0], line[1,1] + line[0,1]])
		self.J = np.sqrt(np.dot(self.F, self.F))
		nor = np.array([self.F[1], -self.F[0]])
		self.nor = nor/np.linalg.norm(nor)

	def Transform(self, xi):
		return np.dot(self.F, xi) + self.c 

	def Jacobian(self, xi):
		return self.J 

	def F(self, xi):
		return self.F 

	def Finv(self, xi):
		return self.Finv 

	def Normal(self, xi):
		return self.nor 

class FaceInfo:
	def __init__(self, els, iptrans, trans, ftrans, orient, fno=-1):
		self.ElNo1 = els[0]
		self.ipt1 = iptrans[0]
		self.trans1 = trans[0] 
		self.f1 = orient[0]
		if (len(els)==1):
			self.boundary = True 
			self.ElNo2 = els[0] 
			self.ipt2 = iptrans[0] 
			self.trans2 = trans[0] 
			self.f2 = self.f1 
		else:
			self.boundary = False 
			self.ElNo2 = els[1]
			self.ipt2 = iptrans[1] 
			self.trans2 = trans[1] 
			self.f2 = orient[1]

		self.face = ftrans 
		self.fno = fno 

	def __repr__(self):
		s = 'Face ' + str(self.fno) + ':\n' 
		s += '   {} -> {}\n'.format(self.ElNo1, self.ElNo2)
		s += '   ({:.3f},{:.3f}) -> ({:.3f},{:.3f})\n'.format(
			self.face.line[0,0], self.face.line[0,1], self.face.line[1,0], self.face.line[1,1])
		s += '   nor = ({:.3f},{:.3f})\n'.format(self.face.Normal(0)[0], self.face.Normal(0)[1])
		s += '   bdr = {}\n'.format(self.boundary) 
		s += '   orient = {}, {}\n'.format(self.f1, self.f2)
		return s 