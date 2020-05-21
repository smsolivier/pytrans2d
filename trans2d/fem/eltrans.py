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
		self.hx = np.sqrt((box[1,0] - box[0,0])**2 + (box[1,1] - box[0,1])**2)
		self.hy = np.sqrt((box[3,1] - box[1,1])**2 + (box[3,0] - box[1,0])**2)
		self.h = np.array([self.hx/2, self.hy/2])
		self.c = np.array([box[1,0] + box[0,0], box[3,1] + box[1,1]])*.5

		self.j = self.hx*self.hy/4 
		self.f = np.array([[self.hx/2, 0], [0, self.hy/2]])
		self.finv = np.array([[2/self.hx, 0], [0, 2/self.hy]])
		self.finvT = self.finv.transpose()

	def Transform(self, xi):
		return self.c + self.h*xi 

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
		return linalg.Mult(1., gs, self.box) 

	def Finv(self, xi):
		F = self.F(xi)
		return np.linalg.inv(F) 

	def FinvT(self, xi):
		F = self.F(xi)
		return np.linalg.inv(F).transpose()

	def Jacobian(self, xi):
		return np.linalg.det(self.F(xi))

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