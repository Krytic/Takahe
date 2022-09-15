from setuptools import setup, Extension
import re

description = 'A python library to evolve binary star systems in time.'

try:
    with open('README.md', 'r', encoding='utf-8') as f:
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

setup(name='takahe',
      license           = 'MIT License',
      version           = metadata['version'],
      description       = description,
      long_description  = long_description,
      author            = metadata['author'],
      author_email      = metadata['email'],
      packages          = ['takahe'],
      zip_safe          = False,
      homepage          = 'https://github.com/Krytic/Takahe',
      install_requires  = ['numpy',
                           'matplotlib',
                           'numba',
                           'pandas',
                           'diffeqpy',
                           'imageio'
                          ]
    )
