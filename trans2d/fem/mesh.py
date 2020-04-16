#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import igraph 

class AffineTrans:
	def __init__(self, box):
		self.box = box 
		self.hx = box[1,0] - box[0,0] 
		self.hy = box[2,1] - box[1,1] 
		self.h = np.array([self.hx/2, self.hy/2])
		self.c = np.array([self.hx/2, self.hy/2])

		self.J = self.hx*self.hy/4 
		self.F = np.array([[self.hx/2, 0], [0, self.hy/2]])
		self.Finv = np.array([[2/self.hx, 0], [0, 2/self.hy]])

	def Transform(self, xi):
		return self.c + self.h*xi 

	def Jacobian(self, xi):
		return self.J 

	def F(self, xi):
		return self.F 

	def Finv(self, xi):
		return self.Finv 

class RectMesh: 
	def __init__(self, Nx, Ny, xb=[1,1]):
		x1d = np.linspace(0, xb[0], Nx+1)
		y1d = np.linspace(0, xb[1], Ny+1)

		x, y = np.meshgrid(x1d, y1d)
		X = x.flatten()
		Y = y.flatten()

		self.nodes = np.zeros((len(X), 2))
		self.nodes[:,0] = X 
		self.nodes[:,1] = Y
		self.Nn = len(X) 
		self.ele = np.zeros((Nx*Ny, 4), dtype=int) 
		self.Ne = Nx*Ny

		# self.bnodes = np.arange(1, Nx).tolist() + np.arange(Ny*(Nx+1)+1, Ny*(Nx+1)+Nx).tolist() \
		# 	+ np.arange(0, Ny*(Nx+1)+1, Nx+1).tolist() + np.arange(Nx, Nx+Ny*(Nx+1)+1, Nx+1).tolist()
		# self.bnodes.sort()

		e = 0 
		Nnx = Nx+1
		Nny = Ny+1
		for i in range(Ny):
			for j in range(Nx):
				self.ele[e,0] = i*Nnx + j 
				self.ele[e,1] = i*Nnx + j + 1 
				self.ele[e,2] = (i+1)*Nnx + j + 1 
				self.ele[e,3] = (i+1)*Nnx + j 

				e += 1 

		self.graph = igraph.Graph()
		self.graph.add_vertices(self.Ne)
		for i in range(Ny):
			for j in range(Nx):
				e = j + i*Nx 
				edges = []
				if (j<Nx-1):
					edges.append((e,e+1))
				if (i<Ny-1):
					edges.append((e,e+Nx))

				self.graph.add_edges(edges)

		bseq = self.graph.vs(_degree_lt=4)
		self.bel = [i.index for i in bseq] # elements on boundary 

		# loop over edges to build face transformations 

		# build transformations 
		self.trans = []
		for e in range(self.Ne):
			self.trans.append(AffineTrans(self.nodes[self.ele[e]]))

	def WriteVTK(self, fname, point=None, cell=None):
		''' plot discontinuous (ie shared nodes are redundantly defined) ''' 
		f = open(fname + '.vtk', 'w') 
		f.write('# vtk DataFile Version 2.0\nMesh\nASCII\nDATASET UNSTRUCTURED_GRID\n')
		Nn = np.shape(self.nodes)[0] 
		f.write('POINTS {} float\n'.format(self.Ne*4)) 
		for e in range(self.Ne):
			for n in range(4):
				node = self.nodes[self.ele[e][n],:]
				f.write('{} {} {}\n'.format(node[0], node[1], 0))

		f.write('CELLS {} {}\n'.format(self.Ne, self.Ne*5)) 
		count = 0 
		for e in range(self.Ne):
			f.write('4 ')
			for n in range(4):
				f.write('{} '.format(count))
				count += 1

			f.write('\n') 

		f.write('CELL_TYPES {}\n'.format(self.Ne)) 
		for e in range(self.Ne):
			f.write('9\n')
	
		if (point!=None):
			f.write('POINT_DATA {}\n'.format(4*self.Ne))
			for key in point:
				data = point[key] 
				if (len(data)==4*self.Ne): # scalar data 
					f.write('SCALARS {} float\n'.format(key))
					f.write('LOOKUP_TABLE default\n')
					for n in range(len(data)):
						f.write('{}\n'.format(data[n])) 
				elif (len(data)==8*self.Ne): # vector data 
					f.write('VECTORS {} float\n'.format(key))
					half = int(len(data)/2)
					for n in range(half):
						f.write('{} {} 0\n'.format(data[n], data[n+half]))
				else:
					print('data for key={} not formatted properly'.format(key))

		if (cell!=None):
			f.write('CELL_DATA {}\n'.format(self.Ne)) 
			for key in cell:
				f.write('SCALARS {} float\n'.format(key)) 
				f.write('LOOKUP_TABLE default\n')
				data = cell[key] 
				for n in range(len(data)):
					f.write('{}\n'.format(data[n])) 