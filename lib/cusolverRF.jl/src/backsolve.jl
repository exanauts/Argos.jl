
function mysv2!(transa::CUSPARSE.SparseChar,
                uplo::CUSPARSE.SparseChar,
                diag::CUSPARSE.SparseChar,
                alpha::Number,
                A::CUSPARSE.CuSparseMatrixCSR{Float64},
                X::CUDA.CuVector{Float64},
                index::CUSPARSE.SparseChar)
    desc = CUSPARSE.CuMatrixDescriptor('G', uplo, diag, index)
    m,n = A.dims
    if m != n
        throw(DimensionMismatch("A must be square, but has dimensions ($m,$n)!"))
    end
    mX = length(X)
    if mX != m
        throw(DimensionMismatch("First dimension of A, $m, and of X, $(size(X)[1]) must match"))
    end
    info = CUSPARSE.csrsv2Info_t[0]
    CUSPARSE.cusparseCreateCsrsv2Info(info)

    out = Ref{Cint}(1)
    CUSPARSE.cusparseDcsrsv2_bufferSize(
            CUSPARSE.handle(), transa, m, nnz(A),
            desc, nonzeros(A), A.rowPtr, A.colVal, info[1],
            out,
    )

    n_bytes = out[]::Cint
    buffer = CUDA.zeros(UInt8, 10 * n_bytes)

    CUSPARSE.cusparseDcsrsv2_analysis(
        CUSPARSE.handle(), transa, m, nnz(A),
        desc, nonzeros(A), A.rowPtr, A.colVal, info[1],
        CUSPARSE.CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer,
    )
    posit = Ref{Cint}(1)
    CUSPARSE.cusparseXcsrsv2_zeroPivot(CUSPARSE.handle(), info[1], posit)

    if posit[] >= 0
        error("Structural/numerical zero in A at ($(posit[]),$(posit[])))")
    end

    CUSPARSE.cusparseDcsrsv2_solve(CUSPARSE.handle(), transa, m,
            nnz(A), alpha, desc, nonzeros(A), A.rowPtr,
            A.colVal, info[1], X, X,
            CUSPARSE.CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)

    CUSPARSE.cusparseDestroyCsrsv2Info(info[1])
    X
end

struct CuSparseBackSV
    transa::CUSPARSE.SparseChar
    descL::CUSPARSE.CuMatrixDescriptor
    descU::CUSPARSE.CuMatrixDescriptor
    infoL::Vector{Ptr{Cvoid}}
    infoU::Vector{Ptr{Cvoid}}
    buffer::CuVector{UInt8}
end

function CuSparseBackSV(
    A::CuSparseMatrixCSR, transa::CUSPARSE.SparseChar,
)
    descL = CUSPARSE.CuMatrixDescriptor('G', 'L', 'U', 'O')
    descU = CUSPARSE.CuMatrixDescriptor('G', 'U', 'N', 'O')
    m, n = A.dims

    infoL = CUSPARSE.csrsv2Info_t[0]
    CUSPARSE.cusparseCreateCsrsv2Info(infoL)

    outL = Ref{Cint}(1)
    CUSPARSE.cusparseDcsrsv2_bufferSize(
            CUSPARSE.handle(), transa, m, nnz(A),
            descL, nonzeros(A), A.rowPtr, A.colVal, infoL[1],
            outL,
    )

    infoU = CUSPARSE.csrsv2Info_t[0]
    CUSPARSE.cusparseCreateCsrsv2Info(infoU)
    outU = Ref{Cint}(1)
    CUSPARSE.cusparseDcsrsv2_bufferSize(
            CUSPARSE.handle(), transa, m, nnz(A),
            descU, nonzeros(A), A.rowPtr, A.colVal, infoU[1],
            outU,
    )

    @assert outL[] == outU[]
    n_bytes = outL[]::Cint
    buffer = CUDA.zeros(UInt8, n_bytes)

    return CuSparseBackSV(transa, descL, descU, infoL, infoU, buffer)
end

function backsolve!(s::CuSparseBackSV, A::CuSparseMatrixCSR, X::CuVector)
    m,n = A.dims
    alpha = 1.0

    # / L
    CUSPARSE.cusparseDcsrsv2_analysis(
        CUSPARSE.handle(), s.transa, m, nnz(A),
        s.descL, nonzeros(A), A.rowPtr, A.colVal, s.infoL[1],
        CUSPARSE.CUSPARSE_SOLVE_POLICY_USE_LEVEL, s.buffer,
    )
    posit = Ref{Cint}(1)
    CUSPARSE.cusparseXcsrsv2_zeroPivot(CUSPARSE.handle(), s.infoL[1], posit)

    if posit[] >= 0
        error("Structural/numerical zero in A at ($(posit[]),$(posit[])))")
    end

    CUSPARSE.cusparseDcsrsv2_solve(CUSPARSE.handle(), s.transa, m,
            nnz(A), alpha, s.descL, nonzeros(A), A.rowPtr,
            A.colVal, s.infoL[1], X, X,
            CUSPARSE.CUSPARSE_SOLVE_POLICY_USE_LEVEL, s.buffer)


    # / U
    CUSPARSE.cusparseDcsrsv2_analysis(
        CUSPARSE.handle(), s.transa, m, nnz(A),
        s.descU, nonzeros(A), A.rowPtr, A.colVal, s.infoU[1],
        CUSPARSE.CUSPARSE_SOLVE_POLICY_USE_LEVEL, s.buffer,
    )
    posit = Ref{Cint}(1)
    CUSPARSE.cusparseXcsrsv2_zeroPivot(CUSPARSE.handle(), s.infoU[1], posit)

    if posit[] >= 0
        error("Structural/numerical zero in A at ($(posit[]),$(posit[])))")
    end

    CUSPARSE.cusparseDcsrsv2_solve(CUSPARSE.handle(), s.transa, m,
            nnz(A), alpha, s.descU, nonzeros(A), A.rowPtr,
            A.colVal, s.infoU[1], X, X,
            CUSPARSE.CUSPARSE_SOLVE_POLICY_USE_LEVEL, s.buffer)

end

struct CuSparseBackSM
    transa::CUSPARSE.SparseChar
    descL::CUSPARSE.CuMatrixDescriptor
    descU::CUSPARSE.CuMatrixDescriptor
    infoL::Vector{Ptr{Cvoid}}
    infoU::Vector{Ptr{Cvoid}}
    buffer::CuVector{UInt8}
end

function CuSparseBackSM(
    A::CuSparseMatrixCSR, transa::CUSPARSE.SparseChar, X::CuMatrix,
)
    descL = CUSPARSE.CuMatrixDescriptor('G', 'L', 'U', 'O')
    descU = CUSPARSE.CuMatrixDescriptor('G', 'U', 'N', 'O')
    m, n = A.dims
    transxy = 'N'
    alpha = 1.0
    nX = size(X, 2)
    ldx = max(1, stride(X, 2))

    infoL = CUSPARSE.csrsm2Info_t[0]
    CUSPARSE.cusparseCreateCsrsm2Info(infoL)

    outL = Ref{Cint}(1)
    # TODO
    CUSPARSE.cusparseDcsrsm2_bufferSize(
            CUSPARSE.handle(), 0, transa, transxy, m, nX, nnz(A),
            alpha, descL, nonzeros(A), A.rowPtr, A.colVal, X, ldx, infoL[1],
            CUSPARSE.CUSPARSE_SOLVE_POLICY_USE_LEVEL,
            outL,
    )

    infoU = CUSPARSE.csrsm2Info_t[0]
    CUSPARSE.cusparseCreateCsrsm2Info(infoU)
    outU = Ref{Cint}(1)
    CUSPARSE.cusparseDcsrsm2_bufferSize(
            CUSPARSE.handle(), 0, transa, transxy, m, nX, nnz(A),
            alpha, descU, nonzeros(A), A.rowPtr, A.colVal, X, ldx, infoU[1],
            CUSPARSE.CUSPARSE_SOLVE_POLICY_USE_LEVEL,
            outU,
    )

    @assert outL[] == outU[]
    n_bytes = outL[]::Cint
    buffer = CUDA.zeros(UInt8, n_bytes)

    return CuSparseBackSM(transa, descL, descU, infoL, infoU, buffer)
end

function backsolve!(s::CuSparseBackSM, A::CuSparseMatrixCSR, X::CuMatrix)
    m,n = A.dims
    alpha = 1.0
    transxy = 'N'
    nX = size(X, 2)
    ldx = max(1, stride(X, 2))

    # / L
    CUSPARSE.cusparseDcsrsm2_analysis(
        CUSPARSE.handle(), 0, s.transa, transxy, m, nX, nnz(A), alpha,
        s.descL, nonzeros(A), A.rowPtr, A.colVal, X, ldx, s.infoL[1],
        CUSPARSE.CUSPARSE_SOLVE_POLICY_USE_LEVEL, s.buffer,
    )
    posit = Ref{Cint}(1)
    CUSPARSE.cusparseXcsrsm2_zeroPivot(CUSPARSE.handle(), s.infoL[1], posit)

    if posit[] >= 0
        error("Structural/numerical zero in A at ($(posit[]),$(posit[])))")
    end

    CUSPARSE.cusparseDcsrsm2_solve(
        CUSPARSE.handle(), 0, s.transa, transxy, m, nX, nnz(A), alpha,
        s.descL, nonzeros(A), A.rowPtr, A.colVal, X, ldx, s.infoL[1],
        CUSPARSE.CUSPARSE_SOLVE_POLICY_USE_LEVEL, s.buffer,
    )


    # / U
    CUSPARSE.cusparseDcsrsm2_analysis(
        CUSPARSE.handle(), 0, s.transa, transxy, m, nX, nnz(A), alpha,
        s.descU, nonzeros(A), A.rowPtr, A.colVal, X, ldx, s.infoU[1],
        CUSPARSE.CUSPARSE_SOLVE_POLICY_USE_LEVEL, s.buffer,
    )
    posit = Ref{Cint}(1)
    CUSPARSE.cusparseXcsrsm2_zeroPivot(CUSPARSE.handle(), s.infoU[1], posit)

    if posit[] >= 0
        error("Structural/numerical zero in A at ($(posit[]),$(posit[])))")
    end

    CUSPARSE.cusparseDcsrsm2_solve(
        CUSPARSE.handle(), 0, s.transa, transxy, m, nX, nnz(A), alpha,
        s.descU, nonzeros(A), A.rowPtr, A.colVal, X, ldx, s.infoU[1],
        CUSPARSE.CUSPARSE_SOLVE_POLICY_USE_LEVEL, s.buffer,
    )
end

