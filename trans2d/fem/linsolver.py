#!/usr/bin/env python3

import numpy as np
import warnings 
import scipy.sparse as sp
import scipy.sparse.linalg as spla 
import pyamg 
import time 

from .. import utils 

class IterativeSolver:
	def __init__(self, itol, maxiter, LOUD=False):
		self.itol = itol
		self.maxiter = maxiter 
		self.LOUD = LOUD

		self.it = 0
		self.space = 3*' '
		self.start = 0 
		self.el = []

	def Callback(self, r):
		self.it += 1 
		self.norm = np.linalg.norm(r)
		self.el.append(time.time() - self.start)
		self.start = time.time()
		if (self.LOUD):
			print(self.space + 'i={:3}, norm={:.3e}, {:.2f} s/iter'.format(self.it, self.norm, self.el[-1]))

	def IsConverged(self):
		return not(self.it==self.maxiter)

	def Cleanup(self, info):
		if (info>0 or self.it==self.maxiter):
			warnings.warn('linear solver not converged. final tol={:.3e}, info={}'.format(self.norm, info), 
				utils.ToleranceWarning, stacklevel=2)
		if (info<0):
			raise RuntimeError('linear solver error. info={}'.format(info))

class BlockLDU(IterativeSolver):
	def __init__(self, itol, maxiter, inner=1, LOUD=False):
		IterativeSolver.__init__(self, itol, maxiter, LOUD)
		self.inner = inner 

	def Solve(self, A, B, C, D, Ainv, S, M, rhs, P=None):
		self.it = 0
		CAinv = C*Ainv 
		AinvB = Ainv*B 
		amg = pyamg.ruge_stuben_solver(S.tocsr())

		def Prec(b):
			if (P!=None):
				b = P.transpose()*b 
			z1 = b[:A.shape[0]]
			z2 = b[A.shape[0]:] - CAinv*z1

			y1 = Ainv*z1
			y2 = amg.solve(z2, maxiter=self.inner)

			x2 = y2.copy()
			x1 = y1 - AinvB*x2 

			s = np.concatenate((x1, x2))
			if (P!=None):
				s = P*s 

			return s

		p2x2 = spla.LinearOperator(M.shape, Prec)
		self.start = time.time()
		x, info = spla.gmres(M, rhs, M=p2x2, tol=0, atol=self.itol,
			maxiter=self.maxiter, callback=self.Callback, callback_type='legacy', restart=None)
		self.Cleanup(info)

		return x 

class GaussSeidel(IterativeSolver):
	def __init__(self, itol, maxiter, LOUD=False):
		IterativeSolver.__init__(self, itol, maxiter, LOUD)
		self.A = None 

	def SetOperator(self, A):
		self.A = A 
		self.L = sp.tril(A,0).tocsr()
		self.U = sp.triu(A,1).tocsr()		

	def Solve(self, b):
		if (self.A==None):
			raise RuntimeError('must call SetOperator before Solve')

		self.it = 0
		x = np.zeros(self.A.shape[0])
		for n in range(self.maxiter):
			x0 = x.copy()
			x = spla.spsolve_triangular(self.L, b - self.U*x0, lower=True)

			norm = np.linalg.norm(x - x0)
			if (norm < self.itol):
				break 

			self.Callback(norm)

		return x 

class SymGaussSeidel(IterativeSolver):
	def __init__(self, itol, maxiter, LOUD=False):
		IterativeSolver.__init__(self, itol, maxiter, LOUD)
		self.A = None

	def SetOperator(self, A):
		self.A = A 
		self.L1 = sp.tril(A,0).tocsr()
		self.U1 = sp.triu(A,1).tocsr()
		self.L2 = sp.tril(A,-1).tocsr()
		self.U2 = sp.triu(A,0).tocsr()

	def Solve(self, b, x0=None):
		if (self.A == None):
			raise RuntimeError('must call SetOperator before Solve')
		self.it = 0 
		if (type(x0)==np.ndarray):
			x = x0.copy() 
		else:
			x = np.zeros(self.A.shape[0])
		for n in range(self.maxiter):
			x0 = x.copy()
			x = spla.spsolve_triangular(self.L1, b - self.U1*x0, lower=True)
			x = spla.spsolve_triangular(self.U2, b - self.L2*x, lower=False)
			norm = np.linalg.norm(x - x0)
			if (norm < self.itol):
				break 

			self.Callback(norm)

		return x 

class Jacobi(IterativeSolver):
	def __init__(self, itol, maxiter, LOUD=False):
		IterativeSolver.__init__(self, itol, maxiter, LOUD)
		self.A = None

	def SetOperator(self, A):
		self.A = A 
		self.D = A.diagonal()
		self.Aoff = A - sp.diags(self.D)

	def Solve(self, b):
		if (self.A==None):
			raise RuntimeError('must call SetOperator before Solve')
		self.it = 0
		x = np.zeros(self.A.shape[0])
		for n in range(self.maxiter):
			x0 = x.copy()
			x = (b - self.Aoff*x0)/self.D 

			norm = np.linalg.norm(x - x0)
			if (norm < self.itol):
				break 

			self.Callback(norm) 

		return x 

class BlockLDURelax(IterativeSolver):
	def __init__(self, itol, maxiter, relax, inner=1, LOUD=False):
		IterativeSolver.__init__(self, itol, maxiter, LOUD)
		self.inner = inner 
		self.relax = relax 
		self.relax.space = 2*self.space

	def Solve(self, A, Ainv, B, C, D, rhs):
		self.it = 0
		M = sp.bmat([[A,B], [C,D]]) 
		CAinv = C*Ainv 
		AinvB = Ainv*B 
		S = D - C*AinvB 
		amg = pyamg.ruge_stuben_solver(S.tocsr())

		def Prec(b):
			z1 = b[:A.shape[0]]
			z2 = b[A.shape[0]:] - CAinv*z1

			y1 = self.relax.Solve(A, z1)
			y2 = amg.solve(z2, maxiter=self.inner)

			x2 = y2.copy()
			x1 = y1 - AinvB*x2 

			return np.concatenate((x1, x2))

		p2x2 = spla.LinearOperator(M.shape, Prec)
		x, info = spla.gmres(M, rhs, M=p2x2, tol=self.itol, maxiter=self.maxiter, callback=self.Callback)

		return x 

class BlockTri(IterativeSolver):
	def __init__(self, itol, maxiter, inner=1, LOUD=False):
		IterativeSolver.__init__(self, itol, maxiter, LOUD)
		self.inner = inner 

	def Solve(self, A, Ainv, B, C, D, rhs):
		self.it = 0 
		M = sp.bmat([[A,B], [C,D]])
		S = D - C*Ainv*B
		amg = pyamg.ruge_stuben_solver(S.tocsr())

		def Prec(b):
			x1 = Ainv*b[:A.shape[0]]
			x2 = amg.solve(b[A.shape[0]:] - C*x1, maxiter=self.inner)
			return np.concatenate((x1, x2))

		p = spla.LinearOperator(M.shape, Prec)
		self.start = time.time()
		x, info = spla.gmres(M, rhs, M=p, tol=self.itol, maxiter=self.maxiter, callback=self.Callback)
		self.Cleanup(info)

		return x 

class BlockDiag(IterativeSolver):
	def __init__(self, itol, maxiter, inner=1, LOUD=False):
		IterativeSolver.__init__(self, itol, maxiter, LOUD)
		self.inner = inner 

	def Solve(self, A, Ainv, B, C, D, rhs):
		self.it = 0 
		M = sp.bmat([[A,B], [C,D]])
		S = D - C*Ainv*B
		amg = pyamg.ruge_stuben_solver(S.tocsr())

		def Prec(b):
			x1 = Ainv*b[:A.shape[0]]
			x2 = amg.solve(b[A.shape[0]:], maxiter=self.inner)
			return np.concatenate((x1, x2))

		p = spla.LinearOperator(M.shape, Prec)
		self.start = time.time()
		x, info = spla.gmres(M, rhs, M=p, tol=self.itol, maxiter=self.maxiter, callback=self.Callback)
		self.Cleanup(info)

		return x 

class AMGSolver(IterativeSolver):
	def __init__(self, itol, maxiter, inner=1, smoother=None, LOUD=False):
		IterativeSolver.__init__(self, itol, maxiter, LOUD)
		self.inner = inner 
		self.smoother = smoother 
		if (self.smoother==None):
			self.smoother = ('gauss_seidel', {'sweep':'symmetric'})
			# self.smoother.space = 6*' '

	def Solve(self, A, Ahat, b, P=None):
		self.it = 0
		amg = pyamg.ruge_stuben_solver(Ahat.tocsr(), presmoother=self.smoother, postsmoother=self.smoother)
		# if (self.smoother!=None):
			# self.smoother.SetOperator(A)
		def prec(x):
			if (P!=None):
				x = P.transpose()*x 
			y = amg.solve(x, maxiter=self.inner)
			# if (self.smoother!=None):
				# y = self.smoother.Solve(y)

			if (P!=None):
				y = P*y 	

			return y 

		Prec = spla.LinearOperator(A.shape, prec)
		# self.start = time.time()
		x, info = spla.gmres(A.tocsc(), b, x0=np.zeros(A.shape[0]), M=Prec, callback=self.Callback, 
			callback_type='legacy', tol=self.itol, atol=0, maxiter=self.maxiter, restart=None)
		# x, info = pyamg.krylov.fgmres(A.tocsc(), b, tol=self.itol, maxiter=A.shape[0], 
			# M=Prec, callback=self.Callback)
		self.Cleanup(info)

		return x

class LUSolver(IterativeSolver):
	def __init__(self, itol, maxiter, smoother=None, LOUD=False):
		IterativeSolver.__init__(self, itol, maxiter, LOUD)
		self.smoother = smoother
		if (self.smoother!=None):
			self.smoother.space = 6*' '

	def Solve(self, A, Ahat, b, proj=None):
		self.it = 0 
		lu = spla.splu(Ahat)
		if (self.smoother!=None):
			self.smoother.SetOperator(A)
		def prec(x):
			if (proj!=None):
				x = proj.T*x 
			y = lu.solve(x)
			if (self.smoother!=None):
				y = self.smoother.Solve(y) 
			if (proj!=None):
				y = proj*y 

			return y 

		P = spla.LinearOperator(A.shape, prec)
		self.start = time.time()
		# x, info = spla.gmres(A.tocsc(), b, M=P, callback=self.Callback, 
			# callback_type='legacy', tol=self.itol, atol=0, maxiter=self.maxiter, restart=None)
		x, info = pyamg.krylov.gmres(A.tocsc(), b, M=P, callback=self.Callback, tol=self.itol, maxiter=A.shape[0])
		# x, info = spla.cg(A.tocsr(), b, M=P, callback=self.Callback, tol=self.itol, atol=0, maxiter=self.maxiter)
		self.Cleanup(info)

		return x