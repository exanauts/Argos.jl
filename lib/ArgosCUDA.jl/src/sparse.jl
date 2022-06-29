
import CUDA.APIUtils: with_workspace

#=
    SpGEMM
=#
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

    descA = CUSPARSE.CuSparseMatrixDescriptor(A, 'O')
    descB = CUSPARSE.CuSparseMatrixDescriptor(B, 'O')
    descC = CUSPARSE.CuSparseMatrixDescriptor(C, 'O')

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
    CUDA.unsafe_free!(buffer1Size)
    CUDA.unsafe_free!(buffer2Size)
    return C
end


#=
    Utils
=#
function Argos.split_jacobian(J::CuSparseMatrixCSR, nx, nu)
    n, m = size(J)
    @assert m == nx + nu
    Ap, Aj = Vector(J.rowPtr), Vector(J.colVal)
    return Argos._split_jacobian_csr(Ap, Aj, n, nx, nu)
end

function SparseArrays.findnz(J::CuSparseMatrixCSR)
    n, m = size(J)
    Ap, Aj, Az = (J.rowPtr, J.colVal, J.nzVal) .|> Array
    Bi, Bj = Int[], Int[]
    Bz = Float64[]

    for i in 1:n
        for c in Ap[i]:Ap[i+1]-1
            j = Aj[c]
            push!(Bi, i)
            push!(Bj, j)
            push!(Bz, Az[c])
        end
    end
    return Bi, Bj, Bz
end

function Argos.tril_mapping(H::CuSparseMatrixCSR)
    n, m = size(H)
    Ap, Aj = (H.rowPtr, H.colVal) .|> Vector
    csr2tril = Int[]
    k = 1
    @inbounds for i in 1:n
        for c in Ap[i]:Ap[i+1]-1
            j = Aj[c]
            if j <= i
                push!(csr2tril, k)
            end
            k += 1
        end
    end
    return csr2tril |> CuVector
end

function Argos.get_fixed_nnz(J::CuSparseMatrixCSR, ind_fixed, diag_ind)
    n, m = size(J)
    Jp, Jj = Vector(J.rowPtr), Vector(J.colVal)
    return Argos._get_fixed_index_csr(n, m, Jp, Jj, ind_fixed, diag_ind)
end


#=
    CUSOLVERRF
=#
LS.DirectSolver(J::CuSparseMatrixCSR; kwargs...) = LS.DirectSolver(CUSOLVERRF.RFLU(J; kwargs...))
