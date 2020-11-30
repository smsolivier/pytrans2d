#include "Python.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h> 
#include <omp.h>

#define CHECK_FORTRAN

PyObject* SimpleNewOwnData(int nd, npy_intp const* dims, int typenum, void *data) {
	PyObject *ret = PyArray_SimpleNewFromData(nd, dims, typenum, data); 
	PyArray_ENABLEFLAGS(ret, NPY_ARRAY_OWNDATA); 
	return ret; 
}

static PyObject* _polyval(PyObject* self, PyObject* args) {
	PyArrayObject *B; 
	double x; 

	if (!PyArg_ParseTuple(args, "Od", &B, &x)) {
		printf("oof\n"); 
		return NULL; 
	}

	int N = PyArray_DIM(B, 1); 
	int p = PyArray_DIM(B, 0); 

	double *ptr = PyArray_DATA(B); 
	npy_intp dims = {N}; 
	double* shape = malloc(sizeof(double)*N); 
	for (int i=0; i<N; i++) {
		shape[i] = ptr[i]; 
		for (int j=1; j<p; j++) {
			shape[i] = shape[i]*x + ptr[j*N+ i]; 
		}
	}

	return SimpleNewOwnData(1, &dims, NPY_DOUBLE, shape); 
}

static PyObject* _polyval_mult(PyObject* self, PyObject* args) {
	PyArrayObject *B, *x; 
	if (!PyArg_ParseTuple(args, "OO", &B, &x)) return NULL; 

	int nb = PyArray_DIM(B, 1); // number of basis functions 
	int nc = PyArray_DIM(B, 0); // number of coefficients 
	int nx = PyArray_DIM(x, 0); // number of spatial points 

	double* Bptr = PyArray_DATA(B); 
	double* xptr = PyArray_DATA(x); 
	npy_intp dims[2] = {nx, nb}; 
	double* shape = malloc(sizeof(double)*nb*nx); 
	for (int i=0; i<nx; i++) {
		double X = xptr[i]; 
		for (int j=0; j<nb; j++) {
			int ind = i*nb + j; 
			shape[ind] = Bptr[j]; 
			for (int k=1; k<nc; k++) {
				shape[ind] = shape[ind]*X + Bptr[k*nb + j]; 
			}
		}
	}
	return SimpleNewOwnData(2, &dims[0], NPY_DOUBLE, shape); 
}

static PyObject* _polyval2D(PyObject* self, PyObject* args) {
	PyArrayObject *Bx, *By, *X; 
	if (!PyArg_ParseTuple(args, "OOO", &Bx, &By, &X)) return NULL; 

	int nb[2] = {PyArray_DIM(Bx, 1), PyArray_DIM(By, 1)};  
	int nc[2] = {PyArray_DIM(Bx, 0), PyArray_DIM(By, 0)};  

	PyArrayObject *Xd = PyArray_CastToType(X, PyArray_DescrFromType(NPY_FLOAT64), 0); 

	double *Bxp = PyArray_DATA(Bx); 
	double *Byp = PyArray_DATA(By); 
	double *Xp = PyArray_DATA(Xd); 
	npy_intp dims = nb[0]*nb[1]; 
	double *sx = malloc(sizeof(double)*nb[0]); 
	double *sy = malloc(sizeof(double)*nb[1]); 
	for (int i=0; i<nb[0]; i++) {
		sx[i] = Bxp[i]; 
		for (int j=1; j<nc[0]; j++) {
			sx[i] = sx[i]*Xp[0] + Bxp[j*nb[0] + i]; 
		}
	}
	for (int i=0; i<nb[1]; i++) {
		sy[i] = Byp[i]; 
		for (int j=1; j<nc[1]; j++) {
			sy[i] = sy[i]*Xp[1] + Byp[j*nb[1] + i]; 
		}
	}

	double *shape = malloc(sizeof(double)*dims); 
	for (int i=0; i<nb[1]; i++) {
		for (int j=0; j<nb[0]; j++) {
			int ind = j + i*nb[0]; 
			shape[ind] = sx[j] * sy[i]; 
		}
	}
	free(sx); free(sy); 
	return SimpleNewOwnData(1, &dims, NPY_DOUBLE, shape); 
}

static PyObject* _polyvaltp(PyObject *self, PyObject *args) {
	PyArrayObject *C, *X; 
	if (!PyArg_ParseTuple(args, "OO", &C, &X)) return NULL; 

	if (PyArray_NDIM(C)!=3) {
		PyErr_SetString(PyExc_RuntimeError, "must be 3D"); 
		return NULL; 
	}

#ifdef CHECK_FORTRAN
	if (!PyArray_IS_C_CONTIGUOUS(C)) {
		PyErr_SetString(PyExc_RuntimeError, 
			"fortran array passed in"); 
		return NULL; 
	}
#endif

	int npx = PyArray_DIM(C, 0); 
	int npy = PyArray_DIM(C, 1); 
	int nb = PyArray_DIM(C, 2); 
	if (PyArray_NDIM(X)>1) {
		PyErr_SetString(PyExc_NotImplementedError, "can't do that yet"); 
		return NULL; 
	}
	if (PyArray_DIM(X,0) != 2) {
		PyErr_SetString(PyExc_RuntimeError, "must pass in x and y"); 
		return NULL; 
	}
	PyArrayObject *Xd = PyArray_CastToType(X, PyArray_DescrFromType(NPY_FLOAT64), 0); 

	double *Cp = PyArray_DATA(C); 
	double *Xp = PyArray_DATA(Xd); 

	int nt = npy*nb; 

	double *s = malloc(sizeof(double)*nb); 
	double *d = malloc(sizeof(double)*nb); 
	for (int i=0; i<nb; i++) {
		s[i] = Cp[(npx-1)*nt + (npy-1)*nb + i]; 
	}

	for (int i=1; i<npy; i++) {
		for (int j=0; j<nb; j++) {
			s[j] = s[j]*Xp[1] + Cp[(npx-1)*nt + (npy-i-1)*nb + j]; 
		}
	}

	for (int i=1; i<npx; i++) {
		for (int k=0; k<nb; k++) {
			d[k] = Cp[(npx-i-1)*nt + (npy-1)*nb + k]; 
		}
		for (int j=1; j<npy; j++) {
			for (int k=0; k<nb; k++) {
				d[k] = d[k]*Xp[1] + Cp[(npx-i-1)*nt + (npy-j-1)*nb + k]; 
			}
		}
		for (int k=0; k<nb; k++) {
			s[k] = s[k]*Xp[0] + d[k]; 
		}
	}
	free(d); 
	npy_intp dims = nb; 
	return SimpleNewOwnData(1, &dims, NPY_DOUBLE, s); 
}

static PyMethodDef mainMethods[] = {
	{"PolyVal", _polyval, METH_VARARGS, "horner's method"}, 
	{"PolyVal2", _polyval_mult, METH_VARARGS, "horner's method for multiple evaluation locations"}, 
	{"PolyVal2D", _polyval2D, METH_VARARGS, "horner's method for a tensor product 2d polynomial"},
	{"PolyValTP", _polyvaltp, METH_VARARGS, "tensor product coefficient array"},
	{NULL, NULL, 0, NULL}
}; 

static PyModuleDef pv_mod = {PyModuleDef_HEAD_INIT, "horner", "module for Horner's method", -1, mainMethods}; 

PyMODINIT_FUNC PyInit_horner(void) {
	import_array(); 
	return PyModule_Create(&pv_mod); 
}