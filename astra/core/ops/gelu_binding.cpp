#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <math.h>
#include <cuda_runtime.h>

// Wrapper func for pybind11
pybind11::array_t<float> py_gelu(pybind11::array_t<float> input_array) {
    pybind11::buffer_info buf_info = input_array.request();
    float *ptr = static_cast<float *>(buf_info.ptr);
    int size = buf_info.size;

    float *output = new float[size];
    gelu(ptr, output, size);

    // Return a numpy array
    return pybind11::array_t<float>(buf_info.size, output);
}

PYBIND11_MODULE(gelu_module, m) {
    m.def("gelu", &py_gelu, "GELU Function");
}