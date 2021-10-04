# PSCC2022

Code to reproduce the numerical results presented in *A Feasible Reduced Space Method for Real-Time Optimal Power Flow*.

* The power flow solver is implemented in [ExaPF](https://github.com/exanauts/ExaPF.jl/), as well as the custom automatic differentation backend
* The augmented Lagrangian is implemented in [ExaOpt](https://github.com/exanauts/ExaPF-Opt)

As discussed in the paper, the reduced space algorithm solves OPF problems, both in static and real-time settings.
The algorithm runs on the CPU, but is much faster on GPU.

## Installation
This subdirectory is shipped with its own environment, to reproduce the results more easily.
So far the code has been tested with Julia 1.6 and CUDA 11.3 (note that we need CUDA >= 11.3 to use the Bunch-Kaufman triangular solve on the GPU).

To install all the dependencies (careful, this could take a while as by default CUDA.jl is downloading its own artifact for CUDA):
```shell
$ julia --project
julia> using Pkg ; Pkg.instantiate()

```

## Static OPF
The script to run the static OPF is `static_opf.jl`.
In the current directory, you can load all the functions with:
```julia
julia> include("static_opf.jl")

```
Once the functions loaded, you can reproduce the results presented in the paper (Table II, page 6) with
```julia
julia> pscc_solve_static_opf() # take ~ 15 minutes to run
```

## Real-time OPF
The script to run the real-time OPF is `real_time_opf.jl`.
We use the same procedure as before, and first load the functions implemented in the script:
```julia
julia> include("real_time_opf.jl")

```
The time-series for the loads are stored in `data/`. The loads vary along time, and drop suddenly by 20% at time t=2.
Specify an instance with (by default only the data for `case1354pegase` are provided):
```julia
julia> case = "case1354pegase"
```
To compute the reference trajectory with Ipopt, run
```julia
julia> res_ipopt = rto_ref(case)

```

To compute the control trajectory with the real-time OPF algorithm, run
```julia
julia> res_exa = pscc_real_time_opf(case)

```
This function first computes a solution at time `t=0` with the static OPF algorithm (this can take
several minutes). Then, it tracks a suboptimal solution along time using the algorithm presented
in the paper. If the power flow is found infeasible, the algorithm is recovering feasibility by using
a backtracking line-search procedure.

