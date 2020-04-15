from setuptools import setup

description = 'A python library to evolve binary star systems in time.'

try:
	with open('README.md', 'r') as f:
		long_description = f.read()
except FileNotFoundError:
	long_description = description

setup(name='takahe',
	  version='1.0.0alpha',
	  description=description,
	  long_description=long_description,
	  author='Sean Richards',
	  author_email='sric560@aucklanduni.ac.nz',
	  packages=['takahe'],
	  zip_safe=False,
	  install_requires=[
	  	'hoki',
	  	'numpy',
	  	'matplotlib',
	  	'numba'
	  ]
	)
