# Automatically generated using Clang.jl

using CEnum

@cenum cusolverRfResetValuesFastMode_t::UInt32 begin
    CUSOLVERRF_RESET_VALUES_FAST_MODE_OFF = 0
    CUSOLVERRF_RESET_VALUES_FAST_MODE_ON = 1
end

@cenum cusolverRfMatrixFormat_t::UInt32 begin
    CUSOLVERRF_MATRIX_FORMAT_CSR = 0
    CUSOLVERRF_MATRIX_FORMAT_CSC = 1
end

@cenum cusolverRfUnitDiagonal_t::UInt32 begin
    CUSOLVERRF_UNIT_DIAGONAL_STORED_L = 0
    CUSOLVERRF_UNIT_DIAGONAL_STORED_U = 1
    CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_L = 2
    CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_U = 3
end

@cenum cusolverRfFactorization_t::UInt32 begin
    CUSOLVERRF_FACTORIZATION_ALG0 = 0
    CUSOLVERRF_FACTORIZATION_ALG1 = 1
    CUSOLVERRF_FACTORIZATION_ALG2 = 2
end

@cenum cusolverRfTriangularSolve_t::UInt32 begin
    CUSOLVERRF_TRIANGULAR_SOLVE_ALG1 = 1
    CUSOLVERRF_TRIANGULAR_SOLVE_ALG2 = 2
    CUSOLVERRF_TRIANGULAR_SOLVE_ALG3 = 3
end

@cenum cusolverRfNumericBoostReport_t::UInt32 begin
    CUSOLVERRF_NUMERIC_BOOST_NOT_USED = 0
    CUSOLVERRF_NUMERIC_BOOST_USED = 1
end


const cusolverRfCommon = Cvoid
const cusolverRfHandle_t = Ptr{cusolverRfCommon}
const cusolverSpContext = Cvoid
const cusolverSpHandle_t = Ptr{cusolverSpContext}
const csrqrInfo = Cvoid
const csrqrInfo_t = Ptr{csrqrInfo}
const csrluInfoHost = Cvoid
const csrluInfoHost_t = Ptr{csrluInfoHost}
const csrqrInfoHost = Cvoid
const csrqrInfoHost_t = Ptr{csrqrInfoHost}
const csrcholInfoHost = Cvoid
const csrcholInfoHost_t = Ptr{csrcholInfoHost}
const csrcholInfo = Cvoid
const csrcholInfo_t = Ptr{csrcholInfo}

@cenum cusparseSpSVAlg_t::UInt32 begin
    CUSPARSE_SPSV_ALG_DEFAULT = 0
end

const cusparseSpSVDescr = Cvoid
const cusparseSpSVDescr_t = Ptr{cusparseSpSVDescr}

@cenum cusparseSpSMAlg_t::UInt32 begin
    CUSPARSE_SPSM_ALG_DEFAULT = 0
end

const cusparseSpSMDescr = Cvoid
const cusparseSpSMDescr_t = Ptr{cusparseSpSMDescr}

@cenum cusparseSpMatAttribute_t::UInt32 begin
    CUSPARSE_SPMAT_FILL_MODE = 0
    CUSPARSE_SPMAT_DIAG_TYPE = 1
end

