## QuantropyPro
[![Build Status](https://travis-ci.org/uwescience/shablona.svg?branch=master)](https://travis-ci.org/uwescience/shablona)

This project is a Python-built set of programs designed to calculate relative behaviors of energy states for peptide-solvent PDB networks using random matrix theory (RMT). The programs characterize energy state behavior by visualzing the distributions of eigenvalues and eigenvalue spacings to exhibit the Gaussian Orthogonal Ensemble spectral statistics. The programs are GPU-optimized with the CuPy Python library to take advantage of parallel performance for the computationally intensive task of large-tensor multiplication. This optimization will outperform numpy or other non-parallelized libraries exponentially better as input pdb file increases.

This repository includes 5 example programs used for energy-state visulaizations in the example_files folder. These include:

eigenvalue_visual.py: creates probability distributions of calculated energy states associated with network at various interaction cutoff distances and frames

entropy_visual.py: creates visualization of calculated continuous differential entropy associated network at varied frames for set interaction cutoff distance

probdensity_visual.py: creates visulazations of probability distributions of spacings between calculated energy states. It is designed to produce bivariate 3D plots, bivariate heat map gifs, or 2D single-variable dependent plots of probability distribution dependent upon eigenvalue index, frame in simulation, or spacing value.

stdev_visual.py: creates visulazations of standard deviation of spacings between calculated energy states. It is designed to produce bivariate 3D plots or 2D single-variable dependent plots of standard devaition dependent upon eigenvalue index, frame in simulation, or standard deviation value.

### Project Format
  QuantropyPro/
    |- README.md
    |- quantropypro/
       |- __init__.py
       |- quantropypro.py
       |- due.py
       |- data/
          |- ...
       |- tests/
          |- ...
    |- example_files/
       |- ...
    |- doc/
       |- Makefile
       |- conf.py
       |- sphinxext/
          |- ...
       |- _static/
          |- ...
    |- setup.py
    |- .travis.yml
    |- .mailmap
    |- appveyor.yml
    |- LICENSE
    |- Makefile
    |- ipynb/


### Licensing
This project is MIT licensed. Copyright (c) 2021, Christian Frech, Carnegie Mellon University

