#!/usr/bin/env python3

import numpy as np
import igraph 

from .eltrans import AffineTrans, LinearTrans, AffineFaceTrans, FaceInfo

class AbstractMesh: 
	def __init__(self, nodes, ele, order=0):
		self.nodes = nodes 
		self.ele = ele 
		self.order = order 
		self.Nn = nodes.shape[0] 
		self.Ne = ele.shape[0] 

		self.trans = [] 
		for e in range(self.Ne):
			if (order==0):
				self.trans.append(AffineTrans(self.nodes[self.ele[e]], e))
			elif (order==1):
				self.trans.append(LinearTrans(self.nodes[self.ele[e]], e))
			else:
				raise AttributeError('mesh order ' + str(order) + ' not defined') 

		# build graph 
		els_per_node = [[] for i in range(self.Nn)]
		for e in range(self.Ne):
			for n in range(4):
				els_per_node[self.ele[e,n]] += [e]

		self.graph = igraph.Graph()
		self.graph.add_vertices(self.Ne)
		edges = []
		for n in range(self.Nn):
			els = els_per_node[n]
			if (len(els)>1):
				connect = np.zeros((len(els), len(els)), dtype=bool) 
				for e in range(len(els)):
					for ep in range(e+1, len(els)):
						nid1 = self.ele[els[e]]
						nid2 = self.ele[els[ep]]
						c = 0 
						for i in range(len(nid1)):
							if (nid1[i] in nid2):
								c += 1 

						if (c==2):
							edges.append((els[e], els[ep]))

		self.graph.add_edges(edges)
		self.graph = self.graph.simplify()

		bseq = self.graph.vs(_degree_lt=4)
		self.bel = [i.index for i in bseq] # element ids of boundary elements 

		# loop over edges to build interior face transformations 
		self.iface = [] 
		ref_geom = np.array([[-1,-1], [1,-1], [1,1], [-1,1]])
		self.iface2el = np.zeros((self.Ne, 4), dtype=int) - 1
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
				AffineFaceTrans(pline), # transformation of face
				[f1, f2], # face orientations 
				edge.index # face number 
				))
			self.iface2el[s, f1] = edge.index
			self.iface2el[t, f2] = edge.index 

		self.bface = [] 
		self.bface2el = np.zeros((self.Ne, 4), dtype=int) - 1 
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
						[f], 
						bfn
						))
					self.bface2el[e, f] = bfn 
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
				data = point[key].flatten()
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
				data = cell[key] 
				if (len(data.shape)==1):
					f.write('SCALARS {} float\n'.format(key)) 
					f.write('LOOKUP_TABLE default\n')
					for n in range(len(data)):
						f.write('{}\n'.format(data[n])) 
				else:
					f.write('VECTORS {} float\n'.format(key))
					for n in range(data.shape[0]):
						f.write('{} {} 0\n'.format(data[n,0], data[n,1]))

class RectMesh(AbstractMesh): 
	def __init__(self, Nx, Ny, xb=[1,1]):
		self.order = 0
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
			self.trans.append(AffineTrans(self.nodes[self.ele[e]], e))

		# build graph
		self.graph = igraph.Graph()
		self.graph.add_vertices(self.Ne)
		edges = []
		for i in range(Ny):
			for j in range(Nx):
				e = j + i*Nx 
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
		self.iface2el = np.zeros((self.Ne, 4), dtype=int) - 1
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
				AffineFaceTrans(pline), # transformation of face
				[f1, f2], # face orientations 
				edge.index # face number 
				))
			self.iface2el[s, f1] = edge.index
			self.iface2el[t, f2] = edge.index 

		self.bface = [] 
		self.bface2el = np.zeros((self.Ne, 4), dtype=int) - 1 
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
						[f], 
						bfn
						))
					self.bface2el[e, f] = bfn 
					bfn += 1 