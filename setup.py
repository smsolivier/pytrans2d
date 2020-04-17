#!/usr/bin/env python3

import setuptools

class get_numpy_include(object):
	def __str__(self):
		import numpy 
		return numpy.get_include()

horner = setuptools.Extension('trans2d.ext.horner', 
	sources=['trans2d/ext/horner.c'], 
	include_dirs=[get_numpy_include()])
linalg = setuptools.Extension('trans2d.ext.linalg', 
	sources=['trans2d/ext/linalg.c'], 
	include_dirs=[get_numpy_include()])

setuptools.setup(
	name='trans2d', 
	author='Samuel Olivier', 
	description='high order finite transport methods in 2D', 
	packages=['trans2d', 'trans2d.ext', 'trans2d.fem', 'trans2d.transport'], 
	classifiers=[
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: MIT License",
		"Operating System :: OS Independent"
		],
	python_requires=">=3.6", 
	install_requires=['numpy', 'scipy', 'termcolor', 'pyamg', 'matplotlib', 'quadpy', 'pathlib', 'python-igraph'], 
	ext_modules=[horner, linalg], 
	include_package_data=True
	)
