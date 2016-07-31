# best-arm-identification

How to proceed ?

Choose the parameters of your bandit problem in mainBAI.jl before running this file. 
Experiments will be run in parallel if you open julia with the command julia -p x, where x is (smaller than) the number of CPUs on your machine. 
- choosing typeExp = "Save" will save the results in the results folder 
- results will be displayed in the command window anyways

If you have saved results, running viewResults.jl will help visualizing them (histogram of the number of draws will be printed)
Name and parameters, specified at the beginning of this file, should match with your saved data
Also, you may want to change the histogramm parameters depending on your problem

BAIalgos.jl contains all algorithms: before including it, the type of distribution for the arms should be specified
typeDistribution can take the values "Bernoulli" "Gaussian" "Poisson" "Exponential"

# Configuration information

Experiments were run with the following Julia install: 

julia> VERSION
v"0.4.0"

julia> Pkg.status()
 - Distributions                 0.8.7
 - HDF5                          0.5.6
 - PyPlot                        2.1.1


# MIT License

Copyright (c) 2016 [Aurélien Garivier and Emilie Kaufmann]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
