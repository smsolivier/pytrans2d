import numpy as np 
import pathlib 

class LevelSym:
	def __init__(self, N):
		file = pathlib.Path(__file__).parent.absolute()
		fname = str(file) + '/lsquad/LS_' + str(N) + '.txt'
		try:
			mu, xi, eta, self.w = np.loadtxt(fname, skiprows=1, unpack=True)
		except Exception as e:
			raise ValueError('level symmetric quadrature of order ' + str(N) + ' not available')

		self.N = int(len(mu)/2)
		self.Omega = np.zeros((self.N, 2))
		self.Omega[:,0] = mu[:self.N]
		self.Omega[:,1] = xi[:self.N]
		self.w = self.w[:self.N]*2 
		self.w *= 4*np.pi/np.sum(self.w)
		assert(abs(np.sum(self.w)-4*np.pi)<1e-14) 