from setuptools import setup
from Cython.Build import cythonize

setup(
    name = "bitmex-algo",
    version = "0.0.1",
    url = "none",
    author = "eyeseaevan",
    packages=find_packages(),
    #ext_modules = cythonize("pyClo/pyClo.pyx"),
)