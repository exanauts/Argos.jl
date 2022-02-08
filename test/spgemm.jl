
import CUDA.APIUtils: with_workspace

mutable struct CuSpGEMMDescriptor
    handle::CUSPARSE.cusparseSpGEMMDescr_t

    function CuSpGEMMDescriptor()
        descr_ref = Ref{CUSPARSE.cusparseSpGEMMDescr_t}()
        CUSPARSE.cusparseSpGEMM_createDescr(descr_ref)
        obj = new(descr_ref[])
        finalizer(CUSPARSE.cusparseSpGEMM_destroyDescr, obj)
        obj
    end
end

Base.unsafe_convert(::Type{CUSPARSE.cusparseSpGEMMDescr_t}, desc::CuSpGEMMDescriptor) = desc.handle

function spgemm!(transa::CUSPARSE.SparseChar, transb::CUSPARSE.SparseChar, alpha::Number, A::CuSparseMatrixCSR{T},
             B::CuSparseMatrixCSR{T}, beta::Number, C::CuSparseMatrixCSR{T}, index::CUSPARSE.SparseChar) where {T}
    m,k = size(A)
    n = size(C)[2]

    if transa == 'N' && transb == 'N'
        CUSPARSE.chkmmdims(B,C,k,n,m,n)
    else
        throw(ArgumentError("Sparse mm! only supports transa ($transa) = 'N' and transb ($transb) = 'N'"))
    end

    if beta != zero(beta)
        throw(ArgumentError("Sparse mm! only supports beta=0"))
    end

    descA = CUSPARSE.CuSparseMatrixDescriptor(A)
    descB = CUSPARSE.CuSparseMatrixDescriptor(B)
    descC = CUSPARSE.CuSparseMatrixDescriptor(C)

    spgemm_Desc = CuSpGEMMDescriptor()
    out = Ref{Csize_t}(0)
    CUSPARSE.cusparseSpGEMM_workEstimation(
        CUSPARSE.handle(), transa, transb, Ref{T}(alpha), descA, descB, Ref{T}(beta),
        descC, T, CUSPARSE.CUSPARSE_SPGEMM_DEFAULT, spgemm_Desc, out, CUDA.CU_NULL)

    buffer1Size = CUDA.zeros(UInt8, out[])
    out = Ref{Csize_t}(sizeof(buffer1Size))
    CUSPARSE.cusparseSpGEMM_workEstimation(
        CUSPARSE.handle(), transa, transb, Ref{T}(alpha), descA, descB, Ref{T}(beta),
        descC, T, CUSPARSE.CUSPARSE_SPGEMM_DEFAULT, spgemm_Desc, out, buffer1Size)

    out = Ref{Csize_t}(0)
    CUSPARSE.cusparseSpGEMM_compute(
        CUSPARSE.handle(), transa, transb, Ref{T}(alpha), descA, descB, Ref{T}(beta),
        descC, T, CUSPARSE.CUSPARSE_SPGEMM_DEFAULT, spgemm_Desc, out, CUDA.CU_NULL)
    buffer2Size = CUDA.zeros(UInt8, out[])
    out = Ref{Csize_t}(sizeof(buffer2Size))
    CUSPARSE.cusparseSpGEMM_compute(
        CUSPARSE.handle(), transa, transb, Ref{T}(alpha), descA, descB, Ref{T}(beta),
        descC, T, CUSPARSE.CUSPARSE_SPGEMM_DEFAULT, spgemm_Desc, out, buffer2Size)

    Cm   = Ref{Int64}()
    Cn   = Ref{Int64}()
    Cnnz = Ref{Int64}()
    CUSPARSE.cusparseSpMatGetSize(descC, Cm, Cn, Cnnz)
    @assert C.nnz == Cnnz[]
    CUSPARSE.cusparseCsrSetPointers(descC, C.rowPtr, C.colVal, C.nzVal)
    CUDA.@sync CUSPARSE.cusparseSpGEMM_copy(CUSPARSE.handle(), transa, transb, Ref{T}(alpha), descA, descB,
                        Ref{T}(beta), descC, T, CUSPARSE.CUSPARSE_SPGEMM_DEFAULT, spgemm_Desc)
    return C
end
