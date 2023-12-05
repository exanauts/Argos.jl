
function MadNLP.madnlp(evaluator::AbstractNLPEvaluator; reset=true, options...)
    if reset
        reset!(evaluator)
    end
    nlp = OPFModel(evaluator)
    return MadNLP.madnlp(nlp; options...)
end


"""
    AbstractOPFFormulation

Abstract type for OPF formulation.
"""
abstract type AbstractOPFFormulation end

"""
    run_opf(datafile::String, ::AbstractOPFFormulation; options...)

Solve the OPF problem associated to `datafile` using MadNLP
using the formulation `AbstractOPFFormulation` given as input.
The keyword arguments `options...` are passed to MadNLP to
control the resolution.

By default, Argos implements three different formulations for the OPF:
- [`FullSpace()`](@ref): the classical full-space OPF problem, in sparse format
- [`BieglerReduction()`](@ref): the full-space OPF problem, in condensed form
- [`DommelTinney()`](@ref): the reduced-space OPF problem of Dommel & Tinney

## Notes
- the initial position is provided in the input file `datafile`.

"""
function run_opf end

"""
    run_opf_gpu(datafile::String, ::AbstractOPFFormulation; options...)

Solve the OPF problem associated to `datafile` using MadNLP on the GPU.
"""
function run_opf_gpu end

"""
    FullSpace <: AbstractOPFFormulation

The OPF problem formulated in the full-space.
The KKT system writes as a sparse indefinite symmetric matrix.
It is recommended using a Bunch-Kaufman decomposition
to factorize the resulting KKT system (as implemented in
Pardiso or HSL MA27/MA57).

"""
struct FullSpace <: AbstractOPFFormulation end

function run_opf(datafile::String, ::FullSpace; options...)
    flp = FullSpaceEvaluator(datafile)
    model = OPFModel(flp)
    ips = MadNLP.MadNLPSolver(model; options...)
    MadNLP.solve!(ips)
    return ips
end

"""
    BieglerReduction <: AbstractOPFFormulation

Linearize-then-reduce formulation. Exploit the structure
of the power flow equations in the full-space to reduce and
condense the KKT system. The resulting condensed KKT system
is dense, and can be factorized efficiently using a dense
linear solver as Lapack.

The `BieglerReduction` is mathematically equivalent to the [`FullSpace`](@ref) formulation.

"""
struct BieglerReduction <: AbstractOPFFormulation end

function run_opf(datafile::String, ::BieglerReduction; options...)
    flp = FullSpaceEvaluator(datafile)
    model = OPFModel(flp)

    madnlp_options = Dict{Symbol, Any}(options...)
    madnlp_options[:linear_solver] = MadNLP.LapackCPUSolver
    opt_ipm, opt_linear, logger = MadNLP.load_options(; madnlp_options...)

    KKT = Argos.BieglerKKTSystem{Float64, Vector{Int}, Vector{Float64}, Matrix{Float64}}
    ips = MadNLP.MadNLPSolver{Float64, KKT}(model, opt_ipm, opt_linear; logger=logger)
    MadNLP.solve!(ips)
    return ips
end

"""
    DommelTinney <: AbstractOPFFormulation

Reduce-then-linearize formulation. Implement the
reduced-space formulation of Dommel & Tinney.
The `DommelTinney` formulation optimizes only
with relation to the control `u`, and solve the power flow
equations at each iteration to find the corresponding
state `x(u)` satisfying the power flow equations `g(x(u), u) = 0`.
As a result, the dimension of the problem is significantly reduced.

In the reduced-space, the Jacobian `J` and the Hessian `W` are dense.
To avoid blowing-up the memory, the KKT system is condensed
to factorize only the condensed matrix `K = W + Jᵀ D J`,
with `D` a diagonal matrix associated to the scaling
of the constraints.

### References

Dommel, Hermann W., and William F. Tinney. "Optimal power flow solutions." IEEE Transactions on power apparatus and systems 10 (1968): 1866-1876.
"""
struct DommelTinney <: AbstractOPFFormulation end

function run_opf(datafile::String, ::DommelTinney; options...)
    nlp = ReducedSpaceEvaluator(datafile)
    model = OPFModel(nlp)
    madnlp_options = Dict{Symbol, Any}(options...)
    madnlp_options[:kkt_system] = MadNLP.DENSE_CONDENSED_KKT_SYSTEM
    madnlp_options[:linear_solver] = MadNLP.LapackCPUSolver
    madnlp_options[:lapack_algorithm] = MadNLP.CHOLESKY

    ips = MadNLP.MadNLPSolver(model; madnlp_options...)
    MadNLP.solve!(ips)
    return ips
end

