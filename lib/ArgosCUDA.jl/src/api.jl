
# Forgiving function:
function MadNLP._madnlp_unsafe_wrap(vec::CuVector, n, shift=1)
    return vec
end


function run_opf_gpu(datafile::String, ::Argos.FullSpace; options...)
    flp = Argos.FullSpaceEvaluator(datafile; device=CUDABackend())
    model = Argos.OPFModel(Argos.bridge(flp))
    ips = MadNLP.MadNLPSolver(
        model;
        options...
    )
    MadNLP.solve!(ips)
    return ips
end

function run_opf_gpu(datafile::String, ::Argos.BieglerReduction; options...)
    flp = Argos.FullSpaceEvaluator(datafile; device=CUDABackend())
    model = Argos.OPFModel(Argos.bridge(flp))

    madnlp_options = Dict{Symbol, Any}(options...)
    # madnlp_options[:linear_solver] = LapackGPUSolver
    opt_ipm, opt_linear, logger = MadNLP.load_options(; madnlp_options...)

    KKT = Argos.BieglerKKTSystem{Float64, CuVector{Int}, CuVector{Float64}, CuMatrix{Float64}}
    ips = MadNLP.MadNLPSolver{Float64, KKT}(model, opt_ipm, opt_linear; logger=logger)
    MadNLP.solve!(ips)
    return ips
end

function run_opf_gpu(datafile::String, ::Argos.DommelTinney; options...)
    flp = Argos.ReducedSpaceEvaluator(datafile; device=CUDABackend(), nbatch_hessian=256)
    model = Argos.OPFModel(Argos.bridge(flp))

    madnlp_options = Dict{Symbol, Any}(options...)
    # madnlp_options[:linear_solver] = LapackGPUSolver
    madnlp_options[:kkt_system] = MadNLP.DENSE_CONDENSED_KKT_SYSTEM
    # madnlp_options[:inertia_correction_method] = MadNLP.INERTIA_FREE
    madnlp_options[:lapack_algorithm] = MadNLP.CHOLESKY

    opt_ipm, opt_linear, logger = MadNLP.load_options(; madnlp_options...)

    QN = MadNLP.ExactHessian{Float64, CuVector{Float64}}
    KKT = MadNLP.DenseCondensedKKTSystem{Float64, CuVector{Float64}, CuMatrix{Float64}, QN}
    ips = MadNLP.MadNLPSolver{Float64, KKT}(model, opt_ipm, opt_linear; logger=logger)
    MadNLP.solve!(ips)

    return ips
end
