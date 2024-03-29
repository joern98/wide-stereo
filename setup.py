from setuptools import setup, Extension

module = Extension("plane_sweep_ext", sources=["plane_sweep_impl.py"], extra_compile_args=['/openmp'])

setup(
    name='plane_sweep',
    version='0.1',
    author='joern',
    ext_modules=[module]
)
