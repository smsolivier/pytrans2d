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

		self.j = self.hx*self.hy/4 
		self.f = np.array([[self.hx/2, 0], [0, self.hy/2]])
		self.finv = np.array([[2/self.hx, 0], [0, 2/self.hy]])
		self.finvT = self.finv.transpose()

	def Transform(self, xi):
		return self.c + self.h*xi 

	def Jacobian(self, xi):
		return self.j

	def F(self, xi):
		return self.f 

	def Finv(self, xi):
		return self.finv 

	def FinvT(self, xi):
		return self.finvT 

	def InverseMap(self, x):
		return np.dot(self.finv, x - self.c)

class AffineFaceTrans:
	def __init__(self, line):
		self.line = line 
		self.F = np.dot(.5*np.array([-1,1]), line)
		self.c = .5*np.array([line[1,0] + line[0,0], line[1,1] + line[0,1]])
		self.J = np.sqrt(np.dot(self.F, self.F))
		R = np.array([[np.cos(-np.pi/2), -np.sin(-np.pi/2)], [np.sin(-np.pi/2), np.cos(-np.pi/2)]])
		nor = np.dot(R, self.F)
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
	def __init__(self, els, iptrans, trans, ftrans, fno=-1):
		self.ElNo1 = els[0]
		self.ipt1 = iptrans[0]
		self.trans1 = trans[0] 
		if (len(els)==1):
			self.boundary = True 
			self.ElNo2 = els[0] 
			self.ipt2 = iptrans[0] 
			self.trans2 = trans[0] 
		else:
			self.boundary = False 
			self.ElNo2 = els[1]
			self.ipt2 = iptrans[1] 
			self.trans2 = trans[1] 

		self.face = ftrans 
		self.fno = fno 

	def __repr__(self):
		s = 'Face ' + str(self.fno) + ':\n' 
		s += '   {} -> {}\n'.format(self.ElNo1, self.ElNo2)
		s += '   ({:.3f},{:.3f}) -> ({:.3f},{:.3f})\n'.format(
			self.face.line[0,0], self.face.line[0,1], self.face.line[1,0], self.face.line[1,1])
		s += '   nor = ({:.3f},{:.3f})\n'.format(self.face.Normal(0)[0], self.face.Normal(0)[1])
		s += '   bdr = {}\n'.format(self.boundary) 
		return s 

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

		# build transformations 
		self.trans = []
		for e in range(self.Ne):
			self.trans.append(AffineTrans(self.nodes[self.ele[e]]))

		# build graph
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
		self.bel = [i.index for i in bseq] # element ids of boundary elements 

		# loop over edges to build interior face transformations 
		self.iface = [] 
		ref_geom = np.array([[-1,-1], [1,-1], [1,1], [-1,1]])
		for edge in self.graph.es:
			s = edge.source 
			t = edge.target 
			f1, f2 = self.GetOrientation(s,t)

			n = 4
			rline1 = ref_geom[[f1, (f1+1)%n], :]
			rline2 = ref_geom[[(f2+1)%n, f2], :] # rotate 
			nodes = self.ele[s, [f1, (f1+1)%n]]
			pline = self.nodes[nodes, :]

			self.iface.append(FaceInfo(
				[s,t], # element numbers 
				[AffineFaceTrans(rline1), AffineFaceTrans(rline2)], # 1D -> 2D transformations
				[self.trans[s], self.trans[t]], # 2D element transformations 
				AffineFaceTrans(pline), 
				edge.index # face number 
				))

		self.bface = [] 
		bfn = 0 
		for e in self.bel:
			v = self.graph.vs(e)[0]
			# determine face numbers not in graph 
			n = 4
			faceno = np.zeros(n, dtype=bool)
			for neigh in v.neighbors():
				t = neigh.index 
				s_node = self.ele[e]
				t_node = self.ele[t]
				for i in range(n):
					if (s_node[i] in t_node and s_node[(i+1)%n] in t_node): 
						faceno[i] = True
			for f in range(n):
				if not(faceno[f]):
					rline = ref_geom[[f, (f+1)%n], :]
					pline = self.nodes[self.ele[e, [f, (f+1)%n]]]
					self.bface.append(FaceInfo(
						[e], 
						[AffineFaceTrans(rline)], 
						[self.trans[e]], 
						AffineFaceTrans(pline), 
						bfn
						))
					bfn += 1 

	def GetOrientation(self, s, t):
		s_node = self.ele[s] 
		t_node = self.ele[t] 
		n = len(s_node)
		for i in range(n):
			if (s_node[i] in t_node and s_node[(i+1)%n] in t_node): 
				f1 = i

			if (t_node[i] in s_node and t_node[(i+1)%n] in s_node):
				f2 = i

		return f1, f2 

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