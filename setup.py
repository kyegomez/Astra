from setuptools import setup, Extension
from torch.utils import cpp_extension



setup(
    name='astra',
    ext_modules=[cpp_extension.CppExtension('astra', ['lltm.cpp'])],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)