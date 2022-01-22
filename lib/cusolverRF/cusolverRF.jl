module CUSOLVERRF

import LinearAlgebra
using SparseArrays

import KLU

import CUDA
import CUDA: CuPtr, CuVector, CuMatrix, CuArray
import CUDA.CUSPARSE
import CUDA.CUSOLVER
import CUDA.CUBLAS: unsafe_batch, unsafe_strided_batch

# Headers
include("libcusolverRf_common.jl")
include("libcusolverRf_api.jl")
include("utils.jl")

function cusolverRfCreate()
    handle_ref = Ref{cusolverRfHandle_t}()
    cusolverRfCreate(handle_ref)
    return handle_ref[]
end

function cusolverRfFree(handle)
    if handle != C_NULL
        cusolverRfDestroy(handle)
        handle = C_NULL
    end
end

mutable struct RfHandle
    handle::Ptr{cusolverRfHandle_t}
end

function sparse_rf_handle(;
    fast_mode=true, nzero=0.0, nboost=0.0,
    factorization_algo=CUSOLVERRF_FACTORIZATION_ALG0,
    triangular_algo=CUSOLVERRF_TRIANGULAR_SOLVE_ALG1,
)
    # Create handle
    gH = cusolverRfCreate()
    if fast_mode
        cusolverRfSetResetValuesFastMode(gH, CUSOLVERRF_RESET_VALUES_FAST_MODE_ON)
    else
        cusolverRfSetResetValuesFastMode(gH, CUSOLVERRF_RESET_VALUES_FAST_MODE_OFF)
    end
    cusolverRfSetNumericProperties(gH, nzero, nboost)
    cusolverRfSetMatrixFormat(
        gH,
        CUSOLVERRF_MATRIX_FORMAT_CSR,
        CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_L
    )
    cusolverRfSetAlgs(
        gH, factorization_algo, triangular_algo,
    )
    handle = RfHandle(gH)
    finalizer(rf_free!, handle)
    return handle
end

rf_free!(rf::RfHandle) = cusolverRfFree(rf.handle)

Base.unsafe_convert(::Type{cusolverRfHandle_t}, rf::RfHandle) = rf.handle

struct RfHostLU{T}
    nnzA::Cint
    rowsA::Vector{Cint}
    colsA::Vector{Cint}
    valsA::Vector{T}
    nnzL::Cint
    rowsL::Vector{Cint}
    colsL::Vector{Cint}
    valsL::Vector{T}
    nnzU::Cint
    rowsU::Vector{Cint}
    colsU::Vector{Cint}
    valsU::Vector{T}
    P::Vector{Cint}
    Q::Vector{Cint}
end

function RfHostLU(
    A::CUSPARSE.CuSparseMatrixCSR{T, Ti};
    ordering=:AMD, tol=1e-8, check=true,
) where {T, Ti}
    m, n = size(A)
    @assert m == n # only squared matrices are supported
    nnzA = nnz(A)

    # Transfer data to host
    h_rowsA = A.rowPtr |> Vector{Cint}
    h_colsA = A.colVal |> Vector{Cint}
    h_valsA = A.nzVal |> Vector{T}

    # cusolverRf is 0-based
    h_rowsA .-= Cint(1)
    h_colsA .-= Cint(1)
    h_Qreorder = zeros(Cint, n)
    # Create duplicate matrix for reordering
    h_rowsB = copy(h_rowsA)
    h_colsB = copy(h_colsA)
    h_valsB = copy(h_valsA)

    spH = CUSOLVER.sparse_handle()

    # Create matrix descriptor
    desca = CUSPARSE.CuMatrixDescriptor()
    CUSPARSE.cusparseSetMatType(desca, CUSPARSE.CUSPARSE_MATRIX_TYPE_GENERAL)
    CUSPARSE.cusparseSetMatIndexBase(desca, CUSPARSE.CUSPARSE_INDEX_BASE_ZERO)

    # Reordering
    if ordering == :AMD
        CUSOLVER.cusolverSpXcsrsymamdHost(
            spH,
            n, nnzA, desca,
            h_rowsA, h_colsA, h_Qreorder,
        )
    elseif ordering == :MDQ
        CUSOLVER.cusolverSpXcsrsymmdqHost(
            spH,
            n, nnzA, desca,
            h_rowsA, h_colsA, h_Qreorder,
        )
    elseif ordering == :METIS
        CUSOLVER.cusolverSpXcsrmetisndHost(
            spH,
            n, nnzA, desca,
            h_rowsA, h_colsA, C_NULL, h_Qreorder,
        )
    elseif ordering == :RCM
        CUSOLVER.cusolverSpXcsrsymrcmHost(
            spH,
            n, nnzA, desca,
            h_rowsA, h_colsA, h_Qreorder,
        )
    end

    h_mapBfromA = zeros(Cint, nnzA)
    @inbounds for i in 1:nnzA
        h_mapBfromA[i] = i # identity matrix
    end

    # Compute permutation in two steps
    size_perm = Ref{Csize_t}(0)
    CUSOLVER.cusolverSpXcsrperm_bufferSizeHost(
        spH,
        m, n, nnzA, desca,
        h_rowsB, h_colsB, h_Qreorder, h_Qreorder,
        size_perm,
    )

    buffer_cpu = zeros(Cint, size_perm[])
    CUSOLVER.cusolverSpXcsrpermHost(
        spH,
        m, n, nnzA, desca,
        h_rowsB, h_colsB, h_Qreorder, h_Qreorder, h_mapBfromA,
        buffer_cpu,
    )

    # Apply permutation
    h_valsB = h_valsA[h_mapBfromA]

    # LU Factorization
    info = Ref{CUSOLVER.csrqrInfo_t}()
    cusolverSpCreateCsrluInfoHost(info)

    cusolverSpXcsrluAnalysisHost(
        spH,
        m, nnzA, desca,
        h_rowsB, h_colsB, info[],
    )

    size_internal = Ref{Cint}(0)
    size_lu = Ref{Cint}(0)
    cusolverSpDcsrluBufferInfoHost(
        spH,
        n, nnzA, desca,
        h_valsB, h_rowsB, h_colsB,
        info[],
        size_internal, size_lu
    )

    n_bytes = size_lu[] * sizeof(Cint)
    buffer_lu = zeros(Cint, size_lu[])
    pivot_threshold = 1.0

    cusolverSpDcsrluFactorHost(
        spH, n, nnzA, desca,
        h_valsB, h_rowsB, h_colsB,
        info[], pivot_threshold,
        buffer_lu,
    )

    # Check singularity
    if check
        singularity = Ref{Cint}(0)
        cusolverSpDcsrluZeroPivotHost(
            spH, info[], tol, singularity,
        )

        # Check that the matrix is nonsingular
        if singularity[] >= 0
            SingularException(singularity[])
        end
    end

    # Get size of L and U
    pnnzU = Ref{Cint}(0)
    pnnzL = Ref{Cint}(0)
    cusolverSpXcsrluNnzHost(
        spH,
        pnnzL, pnnzU, info[],
    )

    nnzL = pnnzL[]
    nnzU = pnnzU[]

    # Retrieve L and U matrices
    h_Plu = zeros(Cint, m)
    h_Qlu = zeros(Cint, n)

    h_valsL = zeros(nnzL)
    h_rowsL = zeros(Cint, m+1)
    h_colsL = zeros(Cint, nnzL)

    h_valsU = zeros(nnzU)
    h_rowsU = zeros(Cint, m+1)
    h_colsU = zeros(Cint, nnzU)

    # Extract
    cusolverSpDcsrluExtractHost(
        spH,
        h_Plu, h_Qlu,
        desca,
        h_valsL, h_rowsL, h_colsL,
        desca,
        h_valsU, h_rowsU, h_colsU,
        info[],
        buffer_lu,
    )

    h_P = h_Qreorder[h_Plu .+ 1]
    h_Q = h_Qreorder[h_Qlu .+ 1]

    return RfHostLU(
        nnzA, h_rowsA, h_colsA, h_valsA,
        nnzL, h_rowsL, h_colsL, h_valsL,
        nnzU, h_rowsU, h_colsU, h_valsU,
        h_P, h_Q,
    )
end

struct RfLU{T}
    rf::RfHandle
    nrhs::Int
    n::Int
    m::Int
    nnzA::Int
    drowsA::CuVector{Cint}
    dcolsA::CuVector{Cint}
    dP::CuVector{Cint}
    dQ::CuVector{Cint}
    dT::CuVector{T}
end

function rflu(
    A::CUSPARSE.CuSparseMatrixCSR{T, Ti};
    nrhs=1, ordering=:AMD, check=true, fast_mode=true,
    factorization_algo=CUSOLVERRF_FACTORIZATION_ALG0,
    triangular_algo=CUSOLVERRF_TRIANGULAR_SOLVE_ALG1,
) where {T, Ti}
    if nrhs > 1
        error("Currently CusolverRF supports only one right-hand side.")
    end
    n, m = size(A)
    lu_host = RfHostLU(A; ordering=ordering, check=check)

    # Allocations (device)
    d_T = CUDA.zeros(Cdouble, m * nrhs)

    rf = sparse_rf_handle(;
        fast_mode=fast_mode,
        factorization_algo=factorization_algo,
        triangular_algo=triangular_algo,
    )

    # Assemble internal data structures
    cusolverRfSetupHost(
        n, lu_host.nnzA, lu_host.rowsA, lu_host.colsA, lu_host.valsA,
        lu_host.nnzL, lu_host.rowsL, lu_host.colsL, lu_host.valsL,
        lu_host.nnzU, lu_host.rowsU, lu_host.colsU, lu_host.valsU,
        lu_host.P, lu_host.Q,
        rf
    )
    # Analyze available parallelism
    cusolverRfAnalyze(rf)
    # LU factorization
    cusolverRfRefactor(rf)

    return RfLU{T}(
        rf, nrhs, n, m, lu_host.nnzA,
        lu_host.rowsA, lu_host.colsA, lu_host.P, lu_host.Q, d_T
    )
end

# Update factorization inplace
function rf_refactor!(rflu::RfLU{T}, A::CUSPARSE.CuSparseMatrixCSR{T, Ti}) where {T, Ti}
    cusolverRfResetValues(
        rflu.n, rflu.nnzA,
        rflu.drowsA, rflu.dcolsA, A.nzVal, rflu.dP, rflu.dQ,
        rflu.rf
    )
    cusolverRfRefactor(rflu.rf)
    return
end

# Solve system Ax = b
function rf_solve!(rflu, x::CuVector)
    n = rflu.n
    cusolverRfSolve(rflu.rf, rflu.dP, rflu.dQ, rflu.nrhs, rflu.dT, n, x, n)
    return
end

function rf_extract_factors_host(rflu::RfLU, n)
    pMp = Ptr{Cint}[Ptr{Cint}(0)]
    pMj = Ptr{Cint}[Ptr{Cint}(0)]
    pMx = Ptr{Float64}[Ptr{Float64}(0)]
    pnnzM = Ref{Cint}(0)
    CUSOLVERRF.cusolverRfExtractBundledFactorsHost(
        rflu.rf, pnnzMM, pMp, pMj, pMx
    )
    nnzM = pnnzMM[]
    Mp = unsafe_wrap(Vector{Cint}, pMp[1], n+1)
    Mj = unsafe_wrap(Vector{Cint}, pMj[1], nnzM)
    Mx = unsafe_wrap(Vector{Float64}, pMx[1], nnzM)
    # Julia is 1-indexed
    Mp .+= Cint(1)
    Mj .+= Cint(1)
    return SparseMatrixCSC(n, n, Mp, Mj, Mx)
end

function rf_extract_factors(rflu::RfLU, n)
    pMp = CuPtr{Cint}[CuPtr{Cint}(0)]
    pMj = CuPtr{Cint}[CuPtr{Cint}(0)]
    pMx = CuPtr{Float64}[CuPtr{Float64}(0)]
    pnnzM = Ref{Cint}(0)
    CUSOLVERRF.cusolverRfAccessBundledFactorsDevice(
        rflu.rf, pnnzM, pMp, pMj, pMx
    )
    nnzM = Int(pnnzM[])
    Mp = unsafe_wrap(CuVector{Cint}, pMp[1], n+1)
    Mj = unsafe_wrap(CuVector{Cint}, pMj[1], nnz_)
    Mx = unsafe_wrap(CuVector{Float64}, pMx[1], nnz_)
    # Avoid side effect by copying the indexings
    myMp = copy(Mp)
    myMj = copy(Mj)
    # Julia is 1-indexed
    myMp .+= Cint(1)
    myMj .+= Cint(1)
    return CuSparseMatrixCSR(myMp, myMj, Mx, (n, n))
end

# Batch factorization should not mix with classical LU factorization.
# We implement a structure apart.
struct RfBatchLU{T}
    rf::RfHandle
    batchsize::Int
    n::Int
    m::Int
    nnzA::Int
    drowsA::CuVector{Cint}
    dcolsA::CuVector{Cint}
    dP::CuVector{Cint}
    dQ::CuVector{Cint}
    dT::CuVector{T}
end

function RfBatchLU(
    A::CUSPARSE.CuSparseMatrixCSR{T, Ti}, batchsize::Int;
    ordering=:AMD, check=true, fast_mode=true,
    factorization_algo=CUSOLVERRF_FACTORIZATION_ALG0,
    triangular_algo=CUSOLVERRF_TRIANGULAR_SOLVE_ALG1,
) where {T, Ti}
    n, m = size(A)
    lu_host = RfHostLU(A; ordering=ordering, check=check)

    # Allocations (device)
    d_T = CUDA.zeros(Cdouble, m * batchsize * 2)

    rf = sparse_rf_handle(;
        fast_mode=fast_mode,
        factorization_algo=factorization_algo,
        triangular_algo=triangular_algo,
    )

    # Assemble internal data structures
    h_valsA_batch = Vector{Float64}[lu_host.valsA for i in 1:batchsize]
    ptrA_batch = pointer.(h_valsA_batch)
    cusolverRfBatchSetupHost(
        batchsize,
        n, lu_host.nnzA, lu_host.rowsA, lu_host.colsA, ptrA_batch,
        lu_host.nnzL, lu_host.rowsL, lu_host.colsL, lu_host.valsL,
        lu_host.nnzU, lu_host.rowsU, lu_host.colsU, lu_host.valsU,
        lu_host.P, lu_host.Q,
        rf,
    )
    # Analyze available parallelism
    cusolverRfBatchAnalyze(rf)
    # LU factorization
    cusolverRfBatchRefactor(rf)

    return RfBatchLU{T}(
        rf, batchsize, n, m, lu_host.nnzA,
        lu_host.rowsA, lu_host.colsA, lu_host.P, lu_host.Q, d_T
    )
end

# Update factorization inplace
## Single matrix
function rf_batch_refactor!(rflu::RfBatchLU{T}, A::CUSPARSE.CuSparseMatrixCSR{T, Ti}) where {T, Ti}
    ptrs = [pointer(A.nzVal) for i in 1:rflu.batchsize]
    Aptrs = CuArray(ptrs)
    cusolverRfBatchResetValues(
        rflu.batchsize, rflu.n, rflu.nnzA,
        rflu.drowsA, rflu.dcolsA, Aptrs, rflu.dP, rflu.dQ,
        rflu.rf
    )
    CUDA.unsafe_free!(Aptrs)
    cusolverRfBatchRefactor(rflu.rf)
    return
end
## Multiple matrices
function rf_batch_refactor!(rflu::RfBatchLU{T}, As::Vector{CUSPARSE.CuSparseMatrixCSR{T, Ti}}) where {T, Ti}
    @assert length(As) == rflu.batchsize
    ptrs = [pointer(A.nzVal) for A in As]
    Aptrs = CuArray(ptrs)
    cusolverRfBatchResetValues(
        rflu.batchsize, rflu.n, rflu.nnzA,
        rflu.drowsA, rflu.dcolsA, Aptrs, rflu.dP, rflu.dQ,
        rflu.rf
    )
    CUDA.unsafe_free!(Aptrs)
    cusolverRfBatchRefactor(rflu.rf)
    return
end

function rf_batch_solve!(rflu::RfBatchLU{T}, xs::Vector{CuVector{T}}) where T
    @assert length(xs) == rflu.batchsize
    n, nrhs = rflu.n, 1
    Xptrs = unsafe_batch(xs)
    cusolverRfBatchSolve(rflu.rf, rflu.dP, rflu.dQ, nrhs, rflu.dT, n, Xptrs, n)
    CUDA.unsafe_free!(Xptrs)
    return
end

function rf_batch_solve!(rflu::RfBatchLU{T}, X::CuMatrix{T}) where T
    @assert size(X, 2) == rflu.batchsize
    n = rflu.n
    nrhs = 1
    Xptrs = unsafe_strided_batch(X)
    # Forward and backward solve
    cusolverRfBatchSolve(rflu.rf, rflu.dP, rflu.dQ, nrhs, rflu.dT, n, Xptrs, n)
    CUDA.unsafe_free!(Xptrs)
    return
end

# KLU
function RfLU(
    A::SparseMatrixCSC{Float64, Int32}; fast_mode=false,
    factorization_algo=CUSOLVERRF_FACTORIZATION_ALG2,
    triangular_algo=CUSOLVERRF_TRIANGULAR_SOLVE_ALG2,
)
    n, m = size(A)
    nrhs = 1
    # Allocations (device)
    d_T = CUDA.zeros(Cdouble, m * nrhs)
    # Instantiate cusolverRF
    rf = sparse_rf_handle(;
        fast_mode=fast_mode,
        factorization_algo=factorization_algo,
        triangular_algo=triangular_algo,
    )

    # Initial factorization
    K = KLU.klu(A)
    K.common.scale = 0
    K.common.btf = 0
    K.common.ordering = 0
    K.common.tol = 1e-2
    KLU.klu!(K, A)

    nnzA = nnz(A)
    rowsA, colsA, valsA = convert2csr(A)
    rowsL, colsL, valsL = convert2csr(K.L)
    rowsU, colsU, valsU = convert2csr(K.U)

    rowsL, colsL, valsL = drop_diag_csr(rowsL, colsL, valsL)
    nnzL = length(colsL)
    nnzU = nnz(K.U)
    P = Vector{Cint}(K.p)
    Q = Vector{Cint}(K.q)

    for vals in [rowsA, colsA, rowsL, colsL, rowsU, colsU, P, Q]
        decrement!(vals)
    end
    # Assemble internal data structures
    cusolverRfSetupHost(
        n, nnzA, rowsA, colsA, valsA,
        nnzL, rowsL, colsL, valsL,
        nnzU, rowsU, colsU, valsU,
        P, Q,
        rf
    )
    # Analyze available parallelism
    cusolverRfAnalyze(rf)
    # LU refactorization
    cusolverRfRefactor(rf)

    return RfLU{Cdouble}(
        rf, nrhs, n, m, nnzA,
        rowsA, colsA, P, Q, d_T
    )
end

# Forgiving function
RfLU(A::SparseMatrixCSC{Float64, Int}; kwargs...) = RfLU(SparseMatrixCSC{Float64, Int32}(A); kwargs...)

function RfBatchLU(
    A::SparseMatrixCSC{Float64, Int32}, nbatch; fast_mode=true,
    factorization_algo=CUSOLVERRF_FACTORIZATION_ALG2,
    triangular_algo=CUSOLVERRF_TRIANGULAR_SOLVE_ALG2,
)
    n, m = size(A)
    nrhs = 1
    # Allocations (device)
    d_T = CUDA.zeros(Cdouble, m * nbatch * 2)
    # Instantiate cusolverRF
    rf = sparse_rf_handle(;
        fast_mode=fast_mode,
        factorization_algo=factorization_algo,
        triangular_algo=triangular_algo,
    )

    # Initial factorization
    K = KLU.klu(A)
    K.common.scale = 0
    K.common.btf = 0
    K.common.ordering = 1
    K.common.tol = 1e-1
    KLU.klu!(K, A)

    nnzA = nnz(A)
    rowsA, colsA, valsA = convert2csr(A)
    rowsL, colsL, valsL = convert2csr(K.L)
    rowsU, colsU, valsU = convert2csr(K.U)

    rowsL, colsL, valsL = drop_diag_csr(rowsL, colsL, valsL)
    nnzL = length(colsL)
    nnzU = nnz(K.U)
    P = Vector{Cint}(K.p)
    Q = Vector{Cint}(K.q)

    for vals in [rowsA, colsA, rowsL, colsL, rowsU, colsU, P, Q]
        decrement!(vals)
    end
    # Assemble internal data structures
    h_valsA_batch = Vector{Float64}[valsA for i in 1:nbatch]
    ptrA_batch = pointer.(h_valsA_batch)
    cusolverRfBatchSetupHost(
        nbatch,
        n, nnzA, rowsA, colsA, ptrA_batch,
        nnzL, rowsL, colsL, valsL,
        nnzU, rowsU, colsU, valsU,
        P, Q,
        rf
    )
    # Analyze available parallelism
    cusolverRfBatchAnalyze(rf)
    # LU refactorization
    cusolverRfBatchRefactor(rf)

    return RfBatchLU{Cdouble}(
        rf, nbatch, n, m, nnzA,
        rowsA, colsA, P, Q, d_T
    )
end

RfBatchLU(A::SparseMatrixCSC{Float64, Int}, nbatch; kwargs...) = RfBatchLU(SparseMatrixCSC{Float64, Int32}(A), nbatch; kwargs...)

include("interface.jl")

end
