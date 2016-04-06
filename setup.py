import numpy
from distutils.core import setup
from Cython.Build import cythonize

setup(name='cymandelbulb',
      ext_modules=cythonize("src/*.pyx"),
      extra_compile_args=['-fopenmp'],
      extra_link_args=['-fopenmp'],
      include_dirs=[numpy.get_include()],
      )
