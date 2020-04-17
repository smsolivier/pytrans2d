import numpy as np 
import math

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
		order = math.ceil((p+1)/2)
		assert(order<self.pmax)
		return self.ip_leg2d[order], self.w_leg2d[order]

	def Get1D(self, p):
		order = math.ceil((p+1)/2)
		assert(order<self.pmax)
		return self.ip_leg1d[order-1], self.w_leg1d[order-1]

	def GetFromPoints1D(self, n):
		return self.ip_leg1d[n], self.w_leg1d[n] 

quadrature = Quadrature(20)