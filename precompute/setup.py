from distutils.core import setup,Extension
from Cython.Build import cythonize
import eigency

setup(
    author='nyLiao',
    version='0.0.1',
    install_requires=['Cython>=0.2.15','eigency>=1.77'],
    packages=['little-try'],
    python_requires='>=3',
    ext_modules=cythonize(Extension(
        name='prop',
        sources=['prop.pyx'],
        language='c++',
        extra_compile_args=["-std=c++11", "-O3", "-fopenmp"],
        include_dirs=[".", "module-dir-name"] + eigency.get_includes()[:2] + ["/home/nliao/.local/include/eigen-3.4.0"],
    ))
)
