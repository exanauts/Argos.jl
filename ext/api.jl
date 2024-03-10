
# Forgiving function:
function MadNLP._madnlp_unsafe_wrap(vec::CuVector, n, shift=1)
    return vec
end

function Argos.run_opf_gpu(datafile::String, ::Argos.FullSpace; options...)
    flp = Argos.FullSpaceEvaluator(datafile; device=CUDABackend())
    model = Argos.OPFModel(Argos.bridge(flp))
    solver = MadNLP.MadNLPSolver(
        model;
        kkt_system=MadNLP.SparseKKTSystem,
        callback=MadNLP.SparseCallback,
        options...
    )
    MadNLP.solve!(solver)
    return solver
end

function Argos.run_opf_gpu(datafile::String, ::Argos.BieglerReduction; options...)
    flp = Argos.FullSpaceEvaluator(datafile; device=CUDABackend())
    model = Argos.OPFModel(Argos.bridge(flp))
    KKT = Argos.BieglerKKTSystem{Float64, CuVector{Int}, CuVector{Float64}, CuMatrix{Float64}}
    solver = MadNLP.MadNLPSolver(
        model;
        kkt_system=KKT,
        callback=MadNLP.SparseCallback,
        options...
    )

    MadNLP.solve!(solver)
    return solver
end

function Argos.run_opf_gpu(datafile::String, ::Argos.DommelTinney; options...)
    flp = Argos.ReducedSpaceEvaluator(datafile; device=CUDABackend(), nbatch_hessian=256)
    model = Argos.OPFModel(flp)
    solver = MadNLP.MadNLPSolver(
        model;
        kkt_system=MadNLP.DenseCondensedKKTSystem,
        callback=MadNLP.DenseCallback,
        options...
    )
    MadNLP.solve!(solver)
    return solver
end
