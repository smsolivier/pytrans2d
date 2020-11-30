#define CHECK_FORTRAN

PyObject* SimpleNewOwnData(int nd, npy_intp const* dims, int typenum, void *data) {
	PyObject *ret = PyArray_SimpleNewFromData(nd, dims, typenum, data); 
	PyArray_ENABLEFLAGS(ret, NPY_ARRAY_OWNDATA); 
	return ret; 
}