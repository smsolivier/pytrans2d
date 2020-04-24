import numpy as np 
import warnings 

from .. import fem 
from .. import utils 

warnings.simplefilter('always', category=utils.NegativityWarning)

class QDFactors:
	def __init__(self, space, quad, psi_in=None):
		self.space = space 
		self.quad = quad 
		self.psi_in = psi_in 
		if (psi_in==None):
			self.psi_in = lambda x, Omega: 0 

		self.P = [] 
		for i in range(2):
			for j in range(2):
				self.P.append(fem.GridFunction(space))
		self.phi = fem.GridFunction(space)

	def Compute(self, psi):
		self.psi = psi 
		for i in range(4):
			self.P[i].data *= 0 
		self.phi.data *= 0 

		neg = False
		for a in range(self.quad.N):
			Omega = self.quad.Omega[a]
			w = self.quad.w[a] 
			angle = psi.GetAngle(a)
			for i in range(2):
				for j in range(2):
					self.P[i*2+j].data += w*Omega[i]*Omega[j]*angle.data 
			self.phi.data += w*angle.data 

			if (angle.data<0).any():
				neg = True

		if (neg):
			warnings.warn('negative psi detected', utils.NegativityWarning)

	def EvalTensor(self, trans, xi):
		E = np.zeros((2,2))
		for i in range(2):
			for j in range(2):
				E[i,j] = self.P[2*i+j].Interpolate(trans.ElNo, xi)

		E /= self.phi.Interpolate(trans.ElNo, xi)
		return E 

	def EvalG(self, fi, ip):
		t = 0 
		b = 0 
		for a in range(self.quad.N):
			Omega = self.quad.Omega[a]
			nor = fi.face.Normal(0)
			dot = np.dot(Omega, nor) 
			if (dot>0 or fi.boundary):
				xi = fi.ipt1.Transform(ip)
				elno = fi.ElNo1
			else:
				xi = fi.ipt2.Transform(ip) 
				elno = fi.ElNo2
			psi_at_ip = self.psi.GetAngle(a).Interpolate(elno, xi)
			t += abs(dot) * self.quad.w[a] * psi_at_ip
			b += self.quad.w[a] * psi_at_ip

		return t/b

	def EvalJinBdr(self, fi, ip):
		Jin = 0 
		nor = fi.face.Normal(0)
		xi = fi.ipt1.Transform(ip)
		X = fi.trans1.Transform(xi)
		for a in range(self.quad.N):
			Omega = self.quad.Omega[a]
			dot = np.dot(Omega, nor)
			if (dot<0):
				Jin += dot * self.quad.w[a] * self.psi_in(X,Omega) 

		return Jin 
