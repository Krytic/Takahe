from setuptools import setup, Extension
import re

USE_CYTHON = False

description = 'A python library to evolve binary star systems in time.'

try:
	with open('README.md', 'r') as f:
		long_description = f.read()
except FileNotFoundError:
	long_description = description

metadata = {"version": "",
			"author": "",
			"email": ""
		   }

metadata_file = open("takahe/_metadata.py", "rt").read()

for item in metadata.keys():
	version_regex = rf"^__{item}__ = ['\"]([^'\"]*)['\"]"

	match = re.search(version_regex, metadata_file, re.M)

	if match:
	    metadata[item] = match.group(1)

ext = '.pyx' if USE_CYTHON else '.c'

extensions = [Extension('integrator', ["src/integrator" + ext])]

if USE_CYTHON:
	from Cython.Build import cythonize
	extensions = cythonize(extensions, annotate=True)

setup(name='takahe',
	  version 			= metadata['version'],
	  description 		= description,
	  long_description	= long_description,
	  author 			= metadata['author'],
	  author_email		= metadata['email'],
	  packages			= ['takahe'],
	  zip_safe			= False,
	  ext_modules		= extensions,
	  install_requires	= ['numpy',
	  					   'matplotlib',
	  					   'numba',
	  					   'pandas==1.0.1',
	  					   'diffeqpy'
	  					  ]
	)
