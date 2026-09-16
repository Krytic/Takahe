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
      license='MIT License',
      version=metadata['version'],
      description=description,
      long_description=long_description,
      long_description_content_type='text/markdown',
      author=metadata['author'],
      author_email=metadata['email'],
      packages=['takahe'],
      zip_safe=False,
      project_urls={'Source': 'https://github.com/Krytic/Takahe'},
      classifiers=['Intended Audience :: Science/Research',
                   'License :: OSI Approved :: MIT License',
                   'Programming Language :: Python :: 3.12',
                   'Programming Language :: Python :: 3.13',
                   'Programming Language :: Python :: 3.14',
                   ],
      install_requires=['numpy',
                        'matplotlib',
                        'numba',
                        'scipy',
                        'uncertainties',
                        'tqdm',
                        'glisten'
                        'pandas',
                        'diffeqpy',
                        'imageio'
                        ]
      )
