import numpy as np 
import math
import os 

class Quadrature:
	def __init__(self, pmax):
		self.ip_leg1d = [] 
		self.w_leg1d = [] 
		self.ip_leg2d = [] 
		self.w_leg2d = [] 
		self.pmax = pmax 

		for p in range(1, pmax+1):
			ip, w = np.polynomial.legendre.leggauss(p)
			self.ip_leg1d.append(ip)
			self.w_leg1d.append(w)

			ip2 = [] 
			w2 = []
			for i in range(len(w)):
				for j in range(len(w)):
					ip2.append([ip[i], ip[j]])
					w2.append(w[i]*w[j])

			self.ip_leg2d.append(ip2)
			self.w_leg2d.append(w2)

	def Get(self, p):
		idx = math.ceil((p+1)/2)-1
		assert(idx<self.pmax)
		return self.ip_leg2d[idx], self.w_leg2d[idx]

	def GetLumped(self, el):
		from . import basis 
		if (type(el.basis)==basis.LegendreBasis):
			return self.ip_leg2d[el.p], self.w_leg2d[el.p]
		else:
			raise NotImplementedError('lumped integration of type ' + str(type(el.basis)) + ' not implemented')

	def Get1D(self, p):
		idx = math.ceil((p+1)/2)-1
		assert(idx<self.pmax)
		return self.ip_leg1d[idx], self.w_leg1d[idx]

	def GetFromPoints1D(self, n):
		return self.ip_leg1d[n], self.w_leg1d[n] 

quadrature = Quadrature(int(os.environ['MAX_QORDER'])
	if 'MAX_QORDER' in os.environ else 20)