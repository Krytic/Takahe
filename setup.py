from setuptools import setup

setup(name='takahe',
	  version='1.0.0alpha',
	  description='A python library to evolve binary star systems in time.',
	  author='Sean Richards',
	  author_email='sric560@aucklanduni.ac.nz',
	  license='MIT',
	  packages=['takahe'],
	  zip_safe=False,
	  install_requires=[
	  	'hoki',
	  	'numpy',
	  	'matplotlib'
	  ]
	)