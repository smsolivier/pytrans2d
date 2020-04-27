#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from trans2d import * 

mesh = RectMesh(3,3)
space = H1Space(mesh, LobattoBasis, 4) 
lospace = space.LORefine() 

lospace.mesh.WriteVTK('solution')