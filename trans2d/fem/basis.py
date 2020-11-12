import numpy as np
import quadpy 
from ..ext import linalg

def GenLobatto(p):
	N = p+1 
	rule = quadpy.line_segment.gauss_lobatto(int(N))
	# ip = np.around(rule.points, 14) 
	ip = rule.points 
	V, B, dB = SolveVandermonde(ip)
	return ip, B, dB, V

def GenLagrange(p):
	N = p+1 
	ip = np.linspace(-1,1,N)
	V, B, dB = SolveVandermonde(ip)
	return ip, B, dB, V 

def GenLegendre(p):
	N = p+1 
	ip, w = np.polynomial.legendre.leggauss(N)
	V, B, dB = SolveVandermonde(ip)
	return ip, B, dB, V 

def SolveVandermonde(ip):
	N = len(ip) 
	coef = np.zeros((N,N))
	A = np.zeros((N,N))
	for i in range(N):
		for j in range(N):
			A[i,j] = ip[i]**j 

	coef = np.linalg.solve(A.transpose(), np.eye(N))
	B = np.zeros((N,N))
	dB = np.zeros((N-1,N))

	for i in range(N):
		B[:,i] = coef[i,::-1]
		dB[:,i] = np.polyder(B[:,i])

	return A, B, dB 

class BasisCollection:
	def __init__(self, p, genfun):
		self.p = p 
		self.N = self.p+1 
		self.ip, self.B, self.dB, self.V = genfun(p) 

class LegendreBasis(BasisCollection):
	def __init__(self, p):
		BasisCollection.__init__(self, p, GenLegendre)

class LobattoBasis(BasisCollection):
	def __init__(self, p):
		BasisCollection.__init__(self, p, GenLobatto)

class LagrangeBasis(BasisCollection):
	def __init__(self, p):
		BasisCollection.__init__(self, p, GenLagrange)

class IntervalModal:
	def __init__(self, p, closed):
		lob = closed(p+1)
		self.p = p 
		N = p+1
		self.V = np.zeros((N,N))
		from .quadrature import quadrature
		from . import mesh 
		ip, w = quadrature.Get(2*p+1)

		self.V = np.zeros((N**2, N**2))
		for i in range(N):
			for j in range(N):
				rtrans = mesh.AffineTrans(np.array([
					[lob.ip[j], lob.ip[i]], 
					[lob.ip[j+1], lob.ip[i]], 
					[lob.ip[j], lob.ip[i+1]], 
					[lob.ip[j+1], lob.ip[i+1]]
					]))
				s = np.zeros(N**2)
				for n in range(len(w)):
					xi = rtrans.Transform(ip[n]) 
					s += w[n] * np.outer(xi[0]**np.arange(N), xi[1]**np.arange(N)).flatten() * rtrans.Jacobian(ip[n]) 

				self.V[:,j+i*N] = s 

		B = np.linalg.solve(self.V, np.eye(N**2))
		self.B = np.zeros((N,N,N**2))
		for i in range(N**2):
			self.B[:,:,i] = B[i,:].reshape((N,N))

class MomentBasis:
	def __init__(self, basis):
		self.p = basis.p 
		N = self.p + 1
		self.V = np.zeros((N,N))
		from .quadrature import quadrature 
		from ..ext.horner import PolyVal
		ip, w = quadrature.Get1D(2*self.p+1)
		for n in range(len(w)):
			self.V += np.outer(ip[n]**np.arange(N), PolyVal(basis.B, ip[n])) * w[n] 

		coef = np.linalg.solve(self.V, np.eye(N))
		self.B = np.zeros((N,N))
		self.dB = np.zeros((N-1,N))

		for i in range(N):
			self.B[:,i] = coef[i,::-1]
			self.dB[:,i] = np.polyder(self.B[:,i])

class RTBasis:
	def __init__(self, p):
		self.p = p 
		self.N = 2*(p+1)*(p+2) 
		from .quadrature import quadrature 
		from ..ext.linalg import AddOuter
		qorder = 2*(p+1)+1 
		ip, w = quadrature.Get1D(qorder)
		N = (p+1)*(p+2)
		yrows = [] 
		xrows = []

		ntb = p+1 
		eq = np.zeros((ntb, N))
		for n in range(len(w)):
			s = np.outer(ip[n]**np.arange(p+1), (-1)**np.arange(p+2)).flatten()
			r = ip[n]**np.arange(0,p+1)
			AddOuter(w[n], r, s, eq) 
		yrows.append(eq) 

		eq = np.zeros((ntb, N))
		for n in range(len(w)):
			s = np.outer(ip[n]**np.arange(p+1), 1**np.arange(p+2)).flatten()
			r = ip[n]**np.arange(0,p+1)
			AddOuter(w[n], r, s, eq) 
		yrows.append(eq) 

		eq = np.zeros((ntb, N))
		for n in range(len(w)):
			s = np.outer((-1)**np.arange(p+2), ip[n]**np.arange(p+1)).flatten()
			r = ip[n]**np.arange(0,p+1)
			AddOuter(w[n], r, s, eq) 
		xrows.append(eq) 

		eq = np.zeros((ntb, N))
		for n in range(len(w)):
			s = np.outer(1**np.arange(p+2), ip[n]**np.arange(p+1)).flatten()
			r = ip[n]**np.arange(0,p+1)
			AddOuter(w[n], r, s, eq) 
		xrows.append(eq) 

		if (p>0):
			ip, w = quadrature.Get(qorder)
			ntb = (p+1)*p
			eq = np.zeros((ntb, N))
			for n in range(len(w)):
				s = np.outer(ip[n][0]**np.arange(p+1), ip[n][1]**np.arange(p+2)).flatten()
				r = np.outer(ip[n][0]**np.arange(p+1), ip[n][1]**np.arange(p)).flatten()
				AddOuter(w[n], r, s, eq) 
			yrows.append(eq) 

			eq = np.zeros((ntb, N))
			for n in range(len(w)):
				s = np.outer(ip[n][0]**np.arange(p+2), ip[n][1]**np.arange(p+1)).flatten()
				r = np.outer(ip[n][0]**np.arange(p), ip[n][1]**np.arange(p+1)).flatten()
				AddOuter(w[n], r, s, eq) 
			xrows.append(eq) 

		Ay = np.vstack(yrows) 
		Py = np.zeros((N,N))
		for i in range(p+1):
			Py[i,i] = 1 
		for i in range(p+1):
			Py[i+p+1,(p+1)**2+i] = 1 
		if (p>0):
			for i in range((p+1)*p):
				Py[i+2*(p+1),p+1+i] = 1 

		cy = np.linalg.solve(Ay, Py).transpose()
		self.Cy = np.zeros((p+1, p+2, N))

		Ax = np.vstack(xrows)
		Px = np.zeros((N,N))
		for i in range(p+1):
			Px[i,i*(p+2)] = 1 
		for i in range(p+1):
			Px[p+1+i,p+1+(p+2)*i] = 1 
		if (p>0):
			for i in range(p+1):
				for j in range(p):
					idx = j + i*p
					Px[idx+2*(p+1),1+i*(p+2)+j] = 1 
		cx = np.linalg.solve(Ax, Px).transpose()
		self.Cx = np.zeros((p+2, p+1, N))

		for i in range(N):
			self.Cy[:,:,i] = cy[i,:].reshape((p+1,p+2))
			self.Cx[:,:,i] = cx[i,:].reshape((p+2,p+1))

		self.dCy = np.polynomial.polynomial.polyder(self.Cy, axis=1).copy(order='C')
		self.dCx = np.polynomial.polynomial.polyder(self.Cx, axis=0)