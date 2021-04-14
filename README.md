# place

Pseudo-Linear Approximation to Chaotic Evolution - a Julia library

Currently under development. 

Pseudolinear models can be expressed as a linear sum of nonlinear functions. That is, the nonlinearity is restricted to the individual terms. Radial basis function networks are an example. The code contained here will develop a parsimonious modelling package to build a approxiamtion to the nonlinear evolution operator of a dynamical system from observed time series. 

The Jupyter notebook `buildmodel` contains the current best guess of how to make this work. Working (i.e. it doesn't beep) but not yet extensively tested. Various tweaks, extensions and  experiments remain to be done:
- allow for multi-threaded optimisation (simultaneous model builds and then merge)
- more basis function types
- generate new candidate BF sets subject to existing error at each step within the model growth 
- implement neural nets (should be a single line of code)
- allow for multiple cycles through the model growth process
- nice-ify the output
- local (greedy) optimisation of basis function parameters
- implement full DL penalty (penalise for all basis function parameters) - I probably won't bother to do this, it never seemed to work well in the past
- other stopping criteria? 
- reservoir computing and/or regularisation alternatives
- visualisation of model structure
- compute Lyapunov spectrum from model
- check/correct computation of $\delta$ of the case of 2x2 Q and verify appropriate exit conditions for larger Q
- figure out the difference between a module and a package and `include` and `import` in Julia.

Free for non-commercial use, but please cite the following papers as appropriate - the last one is the source, the rest are tweaks.
- M. Small. Applied Nonlinear Time Series Analysis: Applications in Physics, Physiology and Finance. Nonlinear Science Series A, vol. 52. World Scientific, 2005. (ISBN 981-256-117-X).
- M. Small, K. Judd and A. Mees. "Modeling continuous processes from data." Physical Review E 65 (2002):046704.
- M. Small and C.K. Tse. "Minimum description length neural networks for time series prediction." Physical Review E 66 (2002), 066701.
- M. Small and K. Judd. "Comparison of new nonlinear modeling techniques with applications to infant respiration." Physica D, 117 (1998): 283-298.
- K. Judd and A. Mees. "On selecting models for nonlinear time series" Physica D, 82 (1995): 426-444
