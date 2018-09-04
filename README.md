# Code for the paper *Efficient Gaussian Process Classification Using Polya-Gamma Data Augmentation*

**paper is under review**

### RUN A SIMPLE THE EXAMPLE ###
First install Julia 0.7 following this [link](https://julialang.org/downloads/)
All steps for installing Julia are nicely explained through the help.
To run an example follow this instructions ;
 - Make sure that Jupyter is installed on your machine
 - Run `julia` in your terminal
 - Press `]` and type in `add IJulia` to install IJulia
 - After it's installed press `backspace` and type in `using IJulia; notebook()`.
 - In the Jupyter framework open the file `experiment_example.ipynb` and execute all cells. Please note that, as a first installation it may take some time.

Only two datasets are provided for the example since more preprocessing steps are required for others or because they are simply too large.

### RUN EXPERIMENTS SCRIPT ###

It is unfortunately more complicated to setup everything to run competitors. The procedure to rerun the experiments is:
 - Make sure to have python 3.6 and R > v3.4
 - Install `tensorflow` and `gpflow` through `pip` (`pip install gpflow tensorflow`)
 - The (modified) EP method is the file `EP_stochastic.R` which is self sufficient
 - In Julia type `]` and run `add PyCall RCall ArgParse Distributions StatsBase`
 - If you did not run the example yet, type `add "PATH_TO_THIS_FOLDER/OMGP"`
 - In your terminal go to the `experiment` directory of this folder
 - Run `julia paper_experiments.jl --help` to get all possible commands. Here is a simple example that will run `XGPC` and `SVGPC` on the dataset `Diabetis` with 64 inducing points for 200 iterations with hyperparameter optimization (rest of the parameters will take the default values)
```bash
julia paper_experiments.jl Diabetis --XGPC --SVGPC -M 64 -I 200 -A
```
 - Please note that due to the structure of Julia the first run might be longer due to precompilation times
 - ***Important note:*** If you get an error on loading `gpflow` and `tensorflow`, follow this procedure :
  - in Julia set `ENV["PYTHON"] = "PATH_TO_YOUR_PYTHON_EXECUTABLE/python"`
  - Type `]` then `build PyCall`
  - Relaunch julia and try `using PyCall; @pyimport gpflow` to test if it is working
  - You can find more details on [this link](https://github.com/JuliaPy/PyCall.jl)
 - Some light datasets are already made available in the `data` repository `aXa`,`Bank_marketing`,`Diabetis`,`Electricity`,`German`,`Shuttle`
 - To avoid reloading all the packages everytime, you can also run it in REPL mode my running `julia`, replacing the `args["..."]` variables by what you want and run `include("paper_experiments.jl")`

### CODE DETAILS ###
Our method is part of a larger package called `OMGP.jl`, note that other methods are there either in development or from other papers. One can be interested in the `XGPC_Functions.jl` and `SparseXGPC.jl` files as well as the `OfflineTraining.jl`

All details concerning the experiments are in the files `paper_experiments.jl`,`functions_paper_experiment.jl` and `ind_points_experiments.jl`
