from __future__ import absolute_import, division, print_function
from os.path import join as pjoin

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 1
_version_micro = ''  # use '' for first of series, number for 1 and above
_version_extra = 'dev'
# _version_extra = ''  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))

CLASSIFIERS = ["Development Status :: 1 - Alpha",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: MIT License",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Protien folding prediction"]

# Description should be a one-liner:
description = "Creates visuals of entropy characterizations using quantum-classical hybrid model of peptide folding"
# Long description will go up on the pypi page
long_description = """

QuantropyPro
========
QuantropyPro is a package of python functions which allows the user to analyze
the time evolution of a peptide-solvent system given a PDB file. This process
includes generating adjacency matrices and analyze the Gaussian Orthogonal 
Ensemble spectral statistics and calculate the continuous differnetial entropy
to track the convergence of a system upon energetic equilibrium.

The package includes a class of methods for parsing a PDB file and generating
the atomic and electronic adjacency matrices based upon interatomic
interactions, as well as calculating the associated eigenvalues and spacings,
along with the continuous differential entropy of the system at a given state.
There are also several functions for generating bins and placing values into
bins for probability density distributions. 

As examples of use of the methods, the programs used for the calculation of
figures in the paper "Quantum-Classical Hybrid Model for Peptide Folding 
and Resultant Entropic Behavior" are included with annotations for use.

To get started using these modules in your own projects, please go to the
repository README_.

.. _README: https://github.com/uwescience/shablona/blob/master/README.md


License
=======
``QuantropyPro`` is licensed under the terms of the MIT license. See the file
"LICENSE" for information on the history of this software, terms & conditions
for usage, and a DISCLAIMER OF ALL WARRANTIES.
All trademarks referenced herein are property of their respective holders.
Copyright (c) 2021--, Christian Frech, Carnegie Mellon University
"""

NAME = "QuantropyPro"
MAINTAINER = "Christian Frech"
MAINTAINER_EMAIL = "cfrech@andrew.cmu.edu"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "http://github.com/christianfrech/QuantropyPro"
DOWNLOAD_URL = ""
LICENSE = "MIT"
AUTHOR = "Christian Frech"
AUTHOR_EMAIL = "cfrech@andrew.cmu.edu"
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
PACKAGE_DATA = {'quantropypro': [pjoin('data', '*')]}
REQUIRES = ["numpy", "cupy", "os", "sys", "matplotlib"]
PYTHON_REQUIRES = ">= 3.7"
