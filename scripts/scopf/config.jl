#=
    Config file to run SCOPF benchmark.
=#

# OPF instances
cases = [
    "case118.m",
    # "case300.m",
    # "case1354pegase.m",
    # # "case2869pegase.m",
    # "case9241pegase.m",
    # # "case_ACTIVSg500.m",
    # # "case_ACTIVSg2000.m",
    # # "case_ACTIVSg10k.m",
]

# Source of contingencies (:exadata or :generated)
src_contingencies = :generated

# Maximum contingencies allowable.
max_contingencies = 32

# Number of trial runs to estimate running time.
ntrials = 3

# Save results on disk?
save_results = true

# Should we use the GPU to evaluate the derivatives?
use_gpu = true

# KKT system (FullSpace() or BieglerReduction())
kkt = Argos.BieglerReduction()

# KKT linear solver (Ma27Solver, Ma57Solver, LapackGPUSolver)
linear_solver = LapackGPUSolver

# Verbose level
verbose = true
print_level = if verbose
    MadNLP.DEBUG
else
    MadNLP.ERROR
end

