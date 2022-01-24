# Julia wrapper for header: cusolverRf.h
# Automatically generated using Clang.jl

import CUDA: cuComplex, cuDoubleComplex
import CUDA.CUSOLVER: cusolverStatus_t, libcusolver
import CUDA.CUSPARSE: libcusparse
import CUDA.CUSPARSE: cusparseMatDescr_t, cusparseStatus_t, cusparseHandle_t, cusparseOperation_t, cusparseSpMatDescr_t, cusparseDnVecDescr_t, cusparseDnMatDescr_t
import CUDA: @check
import CUDA.APIUtils: @checked

@checked function cusolverRfCreate(handle)
    ccall((:cusolverRfCreate, libcusolver()), cusolverStatus_t, (Ptr{cusolverRfHandle_t},), handle)
end

@checked function cusolverRfDestroy(handle)
    ccall((:cusolverRfDestroy, libcusolver()), cusolverStatus_t, (cusolverRfHandle_t,), handle)
end

@checked function cusolverRfGetMatrixFormat(handle, format, diag)
    ccall((:cusolverRfGetMatrixFormat, libcusolver()), cusolverStatus_t, (cusolverRfHandle_t, Ptr{cusolverRfMatrixFormat_t}, Ptr{cusolverRfUnitDiagonal_t}), handle, format, diag)
end

@checked function cusolverRfSetMatrixFormat(handle, format, diag)
    ccall((:cusolverRfSetMatrixFormat, libcusolver()), cusolverStatus_t, (cusolverRfHandle_t, cusolverRfMatrixFormat_t, cusolverRfUnitDiagonal_t), handle, format, diag)
end

@checked function cusolverRfSetNumericProperties(handle, zero, boost)
    ccall((:cusolverRfSetNumericProperties, libcusolver()), cusolverStatus_t, (cusolverRfHandle_t, Cdouble, Cdouble), handle, zero, boost)
end

@checked function cusolverRfGetNumericProperties(handle, zero, boost)
    ccall((:cusolverRfGetNumericProperties, libcusolver()), cusolverStatus_t, (cusolverRfHandle_t, Ptr{Cdouble}, Ptr{Cdouble}), handle, zero, boost)
end

@checked function cusolverRfGetNumericBoostReport(handle, report)
    ccall((:cusolverRfGetNumericBoostReport, libcusolver()), cusolverStatus_t, (cusolverRfHandle_t, Ptr{cusolverRfNumericBoostReport_t}), handle, report)
end

@checked function cusolverRfSetAlgs(handle, factAlg, solveAlg)
    ccall((:cusolverRfSetAlgs, libcusolver()), cusolverStatus_t, (cusolverRfHandle_t, cusolverRfFactorization_t, cusolverRfTriangularSolve_t), handle, factAlg, solveAlg)
end

@checked function cusolverRfGetAlgs(handle, factAlg, solveAlg)
    ccall((:cusolverRfGetAlgs, libcusolver()), cusolverStatus_t, (cusolverRfHandle_t, Ptr{cusolverRfFactorization_t}, Ptr{cusolverRfTriangularSolve_t}), handle, factAlg, solveAlg)
end

@checked function cusolverRfGetResetValuesFastMode(handle, fastMode)
    ccall((:cusolverRfGetResetValuesFastMode, libcusolver()), cusolverStatus_t, (cusolverRfHandle_t, Ptr{cusolverRfResetValuesFastMode_t}), handle, fastMode)
end

@checked function cusolverRfSetResetValuesFastMode(handle, fastMode)
    ccall((:cusolverRfSetResetValuesFastMode, libcusolver()), cusolverStatus_t, (cusolverRfHandle_t, cusolverRfResetValuesFastMode_t), handle, fastMode)
end

@checked function cusolverRfSetupHost(n, nnzA, h_csrRowPtrA, h_csrColIndA, h_csrValA, nnzL, h_csrRowPtrL, h_csrColIndL, h_csrValL, nnzU, h_csrRowPtrU, h_csrColIndU, h_csrValU, h_P, h_Q, handle)
    ccall((:cusolverRfSetupHost, libcusolver()), cusolverStatus_t, (Cint, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, cusolverRfHandle_t), n, nnzA, h_csrRowPtrA, h_csrColIndA, h_csrValA, nnzL, h_csrRowPtrL, h_csrColIndL, h_csrValL, nnzU, h_csrRowPtrU, h_csrColIndU, h_csrValU, h_P, h_Q, handle)
end

@checked function cusolverRfSetupDevice(n, nnzA, csrRowPtrA, csrColIndA, csrValA, nnzL, csrRowPtrL, csrColIndL, csrValL, nnzU, csrRowPtrU, csrColIndU, csrValU, P, Q, handle)
    ccall((:cusolverRfSetupDevice, libcusolver()), cusolverStatus_t, (Cint, Cint, CuPtr{Cint}, CuPtr{Cint}, CuPtr{Cdouble}, Cint, CuPtr{Cint}, CuPtr{Cint}, CuPtr{Cdouble}, Cint, CuPtr{Cint}, CuPtr{Cint}, CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint}, cusolverRfHandle_t), n, nnzA, csrRowPtrA, csrColIndA, csrValA, nnzL, csrRowPtrL, csrColIndL, csrValL, nnzU, csrRowPtrU, csrColIndU, csrValU, P, Q, handle)
end

@checked function cusolverRfResetValues(n, nnzA, csrRowPtrA, csrColIndA, csrValA, P, Q, handle)
    ccall((:cusolverRfResetValues, libcusolver()), cusolverStatus_t, (Cint, Cint, CuPtr{Cint}, CuPtr{Cint}, CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint}, cusolverRfHandle_t), n, nnzA, csrRowPtrA, csrColIndA, csrValA, P, Q, handle)
end

@checked function cusolverRfAnalyze(handle)
    ccall((:cusolverRfAnalyze, libcusolver()), cusolverStatus_t, (cusolverRfHandle_t,), handle)
end

@checked function cusolverRfRefactor(handle)
    ccall((:cusolverRfRefactor, libcusolver()), cusolverStatus_t, (cusolverRfHandle_t,), handle)
end

@checked function cusolverRfAccessBundledFactorsDevice(handle, nnzM, Mp, Mi, Mx)
    ccall((:cusolverRfAccessBundledFactorsDevice, libcusolver()), cusolverStatus_t, (cusolverRfHandle_t, Ptr{Cint}, Ptr{CuPtr{Cint}}, Ptr{CuPtr{Cint}}, Ptr{CuPtr{Cdouble}}), handle, nnzM, Mp, Mi, Mx)
end

@checked function cusolverRfExtractBundledFactorsHost(handle, h_nnzM, h_Mp, h_Mi, h_Mx)
    ccall((:cusolverRfExtractBundledFactorsHost, libcusolver()), cusolverStatus_t, (cusolverRfHandle_t, Ptr{Cint}, Ptr{Ptr{Cint}}, Ptr{Ptr{Cint}}, Ptr{Ptr{Cdouble}}), handle, h_nnzM, h_Mp, h_Mi, h_Mx)
end

@checked function cusolverRfExtractSplitFactorsHost(handle, h_nnzL, h_csrRowPtrL, h_csrColIndL, h_csrValL, h_nnzU, h_csrRowPtrU, h_csrColIndU, h_csrValU)
    ccall((:cusolverRfExtractSplitFactorsHost, libcusolver()), cusolverStatus_t, (cusolverRfHandle_t, Ptr{Cint}, Ptr{Ptr{Cint}}, Ptr{Ptr{Cint}}, Ptr{Ptr{Cdouble}}, Ptr{Cint}, Ptr{Ptr{Cint}}, Ptr{Ptr{Cint}}, Ptr{Ptr{Cdouble}}), handle, h_nnzL, h_csrRowPtrL, h_csrColIndL, h_csrValL, h_nnzU, h_csrRowPtrU, h_csrColIndU, h_csrValU)
end

function cusolverRfSolve(handle, P, Q, nrhs, Temp, ldt, XF, ldxf)
    ccall((:cusolverRfSolve, libcusolver()), cusolverStatus_t, (cusolverRfHandle_t, CuPtr{Cint}, CuPtr{Cint}, Cint, CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint), handle, P, Q, nrhs, Temp, ldt, XF, ldxf)
end

@checked function cusolverRfBatchSetupHost(batchSize, n, nnzA, h_csrRowPtrA, h_csrColIndA, h_csrValA_array, nnzL, h_csrRowPtrL, h_csrColIndL, h_csrValL, nnzU, h_csrRowPtrU, h_csrColIndU, h_csrValU, h_P, h_Q, handle)
    ccall((:cusolverRfBatchSetupHost, libcusolver()), cusolverStatus_t, (Cint, Cint, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Ptr{Cdouble}}, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, cusolverRfHandle_t), batchSize, n, nnzA, h_csrRowPtrA, h_csrColIndA, h_csrValA_array, nnzL, h_csrRowPtrL, h_csrColIndL, h_csrValL, nnzU, h_csrRowPtrU, h_csrColIndU, h_csrValU, h_P, h_Q, handle)
end

@checked function cusolverRfBatchResetValues(batchSize, n, nnzA, csrRowPtrA, csrColIndA, csrValA_array, P, Q, handle)
    ccall((:cusolverRfBatchResetValues, libcusolver()), cusolverStatus_t, (Cint, Cint, Cint, CuPtr{Cint}, CuPtr{Cint}, CuPtr{Ptr{Cdouble}}, CuPtr{Cint}, CuPtr{Cint}, cusolverRfHandle_t), batchSize, n, nnzA, csrRowPtrA, csrColIndA, csrValA_array, P, Q, handle)
end

@checked function cusolverRfBatchAnalyze(handle)
    ccall((:cusolverRfBatchAnalyze, libcusolver()), cusolverStatus_t, (cusolverRfHandle_t,), handle)
end

@checked function cusolverRfBatchRefactor(handle)
    ccall((:cusolverRfBatchRefactor, libcusolver()), cusolverStatus_t, (cusolverRfHandle_t,), handle)
end

@checked function cusolverRfBatchSolve(handle, P, Q, nrhs, Temp, ldt, XF_array, ldxf)
    ccall((:cusolverRfBatchSolve, libcusolver()), cusolverStatus_t, (cusolverRfHandle_t, CuPtr{Cint}, CuPtr{Cint}, Cint, CuPtr{Cdouble}, Cint, CuPtr{Ptr{Cdouble}}, Cint), handle, P, Q, nrhs, Temp, ldt, XF_array, ldxf)
end

@checked function cusolverRfBatchZeroPivot(handle, position)
    ccall((:cusolverRfBatchZeroPivot, libcusolver()), cusolverStatus_t, (cusolverRfHandle_t, Ptr{Cint}), handle, position)
end

# Julia wrapper for header: cusolverSp_LOWLEVEL_PREVIEW.h
# Automatically generated using Clang.jl


@checked function cusolverSpCreateCsrluInfoHost(info)
    ccall((:cusolverSpCreateCsrluInfoHost, libcusolver()), cusolverStatus_t, (Ptr{csrluInfoHost_t},), info)
end

@checked function cusolverSpDestroyCsrluInfoHost(info)
    ccall((:cusolverSpDestroyCsrluInfoHost, libcusolver()), cusolverStatus_t, (csrluInfoHost_t,), info)
end

@checked function cusolverSpXcsrluAnalysisHost(handle, n, nnzA, descrA, csrRowPtrA, csrColIndA, info)
    ccall((:cusolverSpXcsrluAnalysisHost, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cint}, Ptr{Cint}, csrluInfoHost_t), handle, n, nnzA, descrA, csrRowPtrA, csrColIndA, info)
end

@checked function cusolverSpScsrluBufferInfoHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
    ccall((:cusolverSpScsrluBufferInfoHost, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cfloat}, Ptr{Cint}, Ptr{Cint}, csrluInfoHost_t, Ptr{Cint}, Ptr{Cint}), handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
end

@checked function cusolverSpDcsrluBufferInfoHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
    ccall((:cusolverSpDcsrluBufferInfoHost, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, csrluInfoHost_t, Ptr{Cint}, Ptr{Cint}), handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
end

@checked function cusolverSpCcsrluBufferInfoHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
    ccall((:cusolverSpCcsrluBufferInfoHost, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{cuComplex}, Ptr{Cint}, Ptr{Cint}, csrluInfoHost_t, Ptr{Cint}, Ptr{Cint}), handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
end

@checked function cusolverSpZcsrluBufferInfoHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
    ccall((:cusolverSpZcsrluBufferInfoHost, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, csrluInfoHost_t, Ptr{Cint}, Ptr{Cint}), handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
end

@checked function cusolverSpScsrluFactorHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pivot_threshold, pBuffer)
    ccall((:cusolverSpScsrluFactorHost, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cfloat}, Ptr{Cint}, Ptr{Cint}, csrluInfoHost_t, Cfloat, Ptr{Cvoid}), handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pivot_threshold, pBuffer)
end

@checked function cusolverSpDcsrluFactorHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pivot_threshold, pBuffer)
    ccall((:cusolverSpDcsrluFactorHost, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, csrluInfoHost_t, Cdouble, Ptr{Cvoid}), handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pivot_threshold, pBuffer)
end

@checked function cusolverSpCcsrluFactorHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pivot_threshold, pBuffer)
    ccall((:cusolverSpCcsrluFactorHost, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{cuComplex}, Ptr{Cint}, Ptr{Cint}, csrluInfoHost_t, Cfloat, Ptr{Cvoid}), handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pivot_threshold, pBuffer)
end

@checked function cusolverSpZcsrluFactorHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pivot_threshold, pBuffer)
    ccall((:cusolverSpZcsrluFactorHost, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, csrluInfoHost_t, Cdouble, Ptr{Cvoid}), handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pivot_threshold, pBuffer)
end

@checked function cusolverSpScsrluZeroPivotHost(handle, info, tol, position)
    ccall((:cusolverSpScsrluZeroPivotHost, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, csrluInfoHost_t, Cfloat, Ptr{Cint}), handle, info, tol, position)
end

@checked function cusolverSpDcsrluZeroPivotHost(handle, info, tol, position)
    ccall((:cusolverSpDcsrluZeroPivotHost, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, csrluInfoHost_t, Cdouble, Ptr{Cint}), handle, info, tol, position)
end

@checked function cusolverSpCcsrluZeroPivotHost(handle, info, tol, position)
    ccall((:cusolverSpCcsrluZeroPivotHost, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, csrluInfoHost_t, Cfloat, Ptr{Cint}), handle, info, tol, position)
end

@checked function cusolverSpZcsrluZeroPivotHost(handle, info, tol, position)
    ccall((:cusolverSpZcsrluZeroPivotHost, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, csrluInfoHost_t, Cdouble, Ptr{Cint}), handle, info, tol, position)
end

@checked function cusolverSpScsrluSolveHost(handle, n, b, x, info, pBuffer)
    ccall((:cusolverSpScsrluSolveHost, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Ptr{Cfloat}, Ptr{Cfloat}, csrluInfoHost_t, Ptr{Cvoid}), handle, n, b, x, info, pBuffer)
end

@checked function cusolverSpDcsrluSolveHost(handle, n, b, x, info, pBuffer)
    ccall((:cusolverSpDcsrluSolveHost, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Ptr{Cdouble}, Ptr{Cdouble}, csrluInfoHost_t, Ptr{Cvoid}), handle, n, b, x, info, pBuffer)
end

@checked function cusolverSpCcsrluSolveHost(handle, n, b, x, info, pBuffer)
    ccall((:cusolverSpCcsrluSolveHost, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Ptr{cuComplex}, Ptr{cuComplex}, csrluInfoHost_t, Ptr{Cvoid}), handle, n, b, x, info, pBuffer)
end

@checked function cusolverSpZcsrluSolveHost(handle, n, b, x, info, pBuffer)
    ccall((:cusolverSpZcsrluSolveHost, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, csrluInfoHost_t, Ptr{Cvoid}), handle, n, b, x, info, pBuffer)
end

@checked function cusolverSpXcsrluNnzHost(handle, nnzLRef, nnzURef, info)
    ccall((:cusolverSpXcsrluNnzHost, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Ptr{Cint}, Ptr{Cint}, csrluInfoHost_t), handle, nnzLRef, nnzURef, info)
end

@checked function cusolverSpScsrluExtractHost(handle, P, Q, descrL, csrValL, csrRowPtrL, csrColIndL, descrU, csrValU, csrRowPtrU, csrColIndU, info, pBuffer)
    ccall((:cusolverSpScsrluExtractHost, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Ptr{Cint}, Ptr{Cint}, cusparseMatDescr_t, Ptr{Cfloat}, Ptr{Cint}, Ptr{Cint}, cusparseMatDescr_t, Ptr{Cfloat}, Ptr{Cint}, Ptr{Cint}, csrluInfoHost_t, Ptr{Cvoid}), handle, P, Q, descrL, csrValL, csrRowPtrL, csrColIndL, descrU, csrValU, csrRowPtrU, csrColIndU, info, pBuffer)
end

@checked function cusolverSpDcsrluExtractHost(handle, P, Q, descrL, csrValL, csrRowPtrL, csrColIndL, descrU, csrValU, csrRowPtrU, csrColIndU, info, pBuffer)
    ccall((:cusolverSpDcsrluExtractHost, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Ptr{Cint}, Ptr{Cint}, cusparseMatDescr_t, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, cusparseMatDescr_t, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, csrluInfoHost_t, Ptr{Cvoid}), handle, P, Q, descrL, csrValL, csrRowPtrL, csrColIndL, descrU, csrValU, csrRowPtrU, csrColIndU, info, pBuffer)
end

@checked function cusolverSpCcsrluExtractHost(handle, P, Q, descrL, csrValL, csrRowPtrL, csrColIndL, descrU, csrValU, csrRowPtrU, csrColIndU, info, pBuffer)
    ccall((:cusolverSpCcsrluExtractHost, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Ptr{Cint}, Ptr{Cint}, cusparseMatDescr_t, Ptr{cuComplex}, Ptr{Cint}, Ptr{Cint}, cusparseMatDescr_t, Ptr{cuComplex}, Ptr{Cint}, Ptr{Cint}, csrluInfoHost_t, Ptr{Cvoid}), handle, P, Q, descrL, csrValL, csrRowPtrL, csrColIndL, descrU, csrValU, csrRowPtrU, csrColIndU, info, pBuffer)
end

@checked function cusolverSpZcsrluExtractHost(handle, P, Q, descrL, csrValL, csrRowPtrL, csrColIndL, descrU, csrValU, csrRowPtrU, csrColIndU, info, pBuffer)
    ccall((:cusolverSpZcsrluExtractHost, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Ptr{Cint}, Ptr{Cint}, cusparseMatDescr_t, Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, cusparseMatDescr_t, Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, csrluInfoHost_t, Ptr{Cvoid}), handle, P, Q, descrL, csrValL, csrRowPtrL, csrColIndL, descrU, csrValU, csrRowPtrU, csrColIndU, info, pBuffer)
end

@checked function cusolverSpCreateCsrqrInfoHost(info)
    ccall((:cusolverSpCreateCsrqrInfoHost, libcusolver()), cusolverStatus_t, (Ptr{csrqrInfoHost_t},), info)
end

@checked function cusolverSpDestroyCsrqrInfoHost(info)
    ccall((:cusolverSpDestroyCsrqrInfoHost, libcusolver()), cusolverStatus_t, (csrqrInfoHost_t,), info)
end

@checked function cusolverSpXcsrqrAnalysisHost(handle, m, n, nnzA, descrA, csrRowPtrA, csrColIndA, info)
    ccall((:cusolverSpXcsrqrAnalysisHost, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, Ptr{Cint}, Ptr{Cint}, csrqrInfoHost_t), handle, m, n, nnzA, descrA, csrRowPtrA, csrColIndA, info)
end

@checked function cusolverSpScsrqrBufferInfoHost(handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
    ccall((:cusolverSpScsrqrBufferInfoHost, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, Ptr{Cfloat}, Ptr{Cint}, Ptr{Cint}, csrqrInfoHost_t, Ptr{Cint}, Ptr{Cint}), handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
end

@checked function cusolverSpDcsrqrBufferInfoHost(handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
    ccall((:cusolverSpDcsrqrBufferInfoHost, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, csrqrInfoHost_t, Ptr{Cint}, Ptr{Cint}), handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
end

@checked function cusolverSpCcsrqrBufferInfoHost(handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
    ccall((:cusolverSpCcsrqrBufferInfoHost, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, Ptr{cuComplex}, Ptr{Cint}, Ptr{Cint}, csrqrInfoHost_t, Ptr{Cint}, Ptr{Cint}), handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
end

@checked function cusolverSpZcsrqrBufferInfoHost(handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
    ccall((:cusolverSpZcsrqrBufferInfoHost, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, csrqrInfoHost_t, Ptr{Cint}, Ptr{Cint}), handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
end

@checked function cusolverSpScsrqrSetupHost(handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, mu, info)
    ccall((:cusolverSpScsrqrSetupHost, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, Ptr{Cfloat}, Ptr{Cint}, Ptr{Cint}, Cfloat, csrqrInfoHost_t), handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, mu, info)
end

@checked function cusolverSpDcsrqrSetupHost(handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, mu, info)
    ccall((:cusolverSpDcsrqrSetupHost, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, Cdouble, csrqrInfoHost_t), handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, mu, info)
end

@checked function cusolverSpCcsrqrSetupHost(handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, mu, info)
    ccall((:cusolverSpCcsrqrSetupHost, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, Ptr{cuComplex}, Ptr{Cint}, Ptr{Cint}, cuComplex, csrqrInfoHost_t), handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, mu, info)
end

@checked function cusolverSpZcsrqrSetupHost(handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, mu, info)
    ccall((:cusolverSpZcsrqrSetupHost, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, cuDoubleComplex, csrqrInfoHost_t), handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, mu, info)
end

@checked function cusolverSpScsrqrFactorHost(handle, m, n, nnzA, b, x, info, pBuffer)
    ccall((:cusolverSpScsrqrFactorHost, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, csrqrInfoHost_t, Ptr{Cvoid}), handle, m, n, nnzA, b, x, info, pBuffer)
end

@checked function cusolverSpDcsrqrFactorHost(handle, m, n, nnzA, b, x, info, pBuffer)
    ccall((:cusolverSpDcsrqrFactorHost, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Cint, Ptr{Cdouble}, Ptr{Cdouble}, csrqrInfoHost_t, Ptr{Cvoid}), handle, m, n, nnzA, b, x, info, pBuffer)
end

@checked function cusolverSpCcsrqrFactorHost(handle, m, n, nnzA, b, x, info, pBuffer)
    ccall((:cusolverSpCcsrqrFactorHost, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Cint, Ptr{cuComplex}, Ptr{cuComplex}, csrqrInfoHost_t, Ptr{Cvoid}), handle, m, n, nnzA, b, x, info, pBuffer)
end

@checked function cusolverSpZcsrqrFactorHost(handle, m, n, nnzA, b, x, info, pBuffer)
    ccall((:cusolverSpZcsrqrFactorHost, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, csrqrInfoHost_t, Ptr{Cvoid}), handle, m, n, nnzA, b, x, info, pBuffer)
end

@checked function cusolverSpScsrqrZeroPivotHost(handle, info, tol, position)
    ccall((:cusolverSpScsrqrZeroPivotHost, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, csrqrInfoHost_t, Cfloat, Ptr{Cint}), handle, info, tol, position)
end

@checked function cusolverSpDcsrqrZeroPivotHost(handle, info, tol, position)
    ccall((:cusolverSpDcsrqrZeroPivotHost, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, csrqrInfoHost_t, Cdouble, Ptr{Cint}), handle, info, tol, position)
end

@checked function cusolverSpCcsrqrZeroPivotHost(handle, info, tol, position)
    ccall((:cusolverSpCcsrqrZeroPivotHost, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, csrqrInfoHost_t, Cfloat, Ptr{Cint}), handle, info, tol, position)
end

@checked function cusolverSpZcsrqrZeroPivotHost(handle, info, tol, position)
    ccall((:cusolverSpZcsrqrZeroPivotHost, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, csrqrInfoHost_t, Cdouble, Ptr{Cint}), handle, info, tol, position)
end

@checked function cusolverSpScsrqrSolveHost(handle, m, n, b, x, info, pBuffer)
    ccall((:cusolverSpScsrqrSolveHost, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, csrqrInfoHost_t, Ptr{Cvoid}), handle, m, n, b, x, info, pBuffer)
end

@checked function cusolverSpDcsrqrSolveHost(handle, m, n, b, x, info, pBuffer)
    ccall((:cusolverSpDcsrqrSolveHost, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Ptr{Cdouble}, Ptr{Cdouble}, csrqrInfoHost_t, Ptr{Cvoid}), handle, m, n, b, x, info, pBuffer)
end

@checked function cusolverSpCcsrqrSolveHost(handle, m, n, b, x, info, pBuffer)
    ccall((:cusolverSpCcsrqrSolveHost, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Ptr{cuComplex}, Ptr{cuComplex}, csrqrInfoHost_t, Ptr{Cvoid}), handle, m, n, b, x, info, pBuffer)
end

@checked function cusolverSpZcsrqrSolveHost(handle, m, n, b, x, info, pBuffer)
    ccall((:cusolverSpZcsrqrSolveHost, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, csrqrInfoHost_t, Ptr{Cvoid}), handle, m, n, b, x, info, pBuffer)
end

@checked function cusolverSpXcsrqrAnalysis(handle, m, n, nnzA, descrA, csrRowPtrA, csrColIndA, info)
    ccall((:cusolverSpXcsrqrAnalysis, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, Ptr{Cint}, Ptr{Cint}, csrqrInfo_t), handle, m, n, nnzA, descrA, csrRowPtrA, csrColIndA, info)
end

@checked function cusolverSpScsrqrBufferInfo(handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
    ccall((:cusolverSpScsrqrBufferInfo, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, Ptr{Cfloat}, Ptr{Cint}, Ptr{Cint}, csrqrInfo_t, Ptr{Cint}, Ptr{Cint}), handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
end

@checked function cusolverSpDcsrqrBufferInfo(handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
    ccall((:cusolverSpDcsrqrBufferInfo, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, csrqrInfo_t, Ptr{Cint}, Ptr{Cint}), handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
end

@checked function cusolverSpCcsrqrBufferInfo(handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
    ccall((:cusolverSpCcsrqrBufferInfo, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, Ptr{cuComplex}, Ptr{Cint}, Ptr{Cint}, csrqrInfo_t, Ptr{Cint}, Ptr{Cint}), handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
end

@checked function cusolverSpZcsrqrBufferInfo(handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
    ccall((:cusolverSpZcsrqrBufferInfo, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, csrqrInfo_t, Ptr{Cint}, Ptr{Cint}), handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
end

@checked function cusolverSpScsrqrSetup(handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, mu, info)
    ccall((:cusolverSpScsrqrSetup, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, Ptr{Cfloat}, Ptr{Cint}, Ptr{Cint}, Cfloat, csrqrInfo_t), handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, mu, info)
end

@checked function cusolverSpDcsrqrSetup(handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, mu, info)
    ccall((:cusolverSpDcsrqrSetup, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, Cdouble, csrqrInfo_t), handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, mu, info)
end

@checked function cusolverSpCcsrqrSetup(handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, mu, info)
    ccall((:cusolverSpCcsrqrSetup, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, Ptr{cuComplex}, Ptr{Cint}, Ptr{Cint}, cuComplex, csrqrInfo_t), handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, mu, info)
end

@checked function cusolverSpZcsrqrSetup(handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, mu, info)
    ccall((:cusolverSpZcsrqrSetup, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, cuDoubleComplex, csrqrInfo_t), handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, mu, info)
end

@checked function cusolverSpScsrqrFactor(handle, m, n, nnzA, b, x, info, pBuffer)
    ccall((:cusolverSpScsrqrFactor, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, csrqrInfo_t, Ptr{Cvoid}), handle, m, n, nnzA, b, x, info, pBuffer)
end

@checked function cusolverSpDcsrqrFactor(handle, m, n, nnzA, b, x, info, pBuffer)
    ccall((:cusolverSpDcsrqrFactor, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Cint, Ptr{Cdouble}, Ptr{Cdouble}, csrqrInfo_t, Ptr{Cvoid}), handle, m, n, nnzA, b, x, info, pBuffer)
end

@checked function cusolverSpCcsrqrFactor(handle, m, n, nnzA, b, x, info, pBuffer)
    ccall((:cusolverSpCcsrqrFactor, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Cint, Ptr{cuComplex}, Ptr{cuComplex}, csrqrInfo_t, Ptr{Cvoid}), handle, m, n, nnzA, b, x, info, pBuffer)
end

@checked function cusolverSpZcsrqrFactor(handle, m, n, nnzA, b, x, info, pBuffer)
    ccall((:cusolverSpZcsrqrFactor, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, csrqrInfo_t, Ptr{Cvoid}), handle, m, n, nnzA, b, x, info, pBuffer)
end

@checked function cusolverSpScsrqrZeroPivot(handle, info, tol, position)
    ccall((:cusolverSpScsrqrZeroPivot, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, csrqrInfo_t, Cfloat, Ptr{Cint}), handle, info, tol, position)
end

@checked function cusolverSpDcsrqrZeroPivot(handle, info, tol, position)
    ccall((:cusolverSpDcsrqrZeroPivot, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, csrqrInfo_t, Cdouble, Ptr{Cint}), handle, info, tol, position)
end

@checked function cusolverSpCcsrqrZeroPivot(handle, info, tol, position)
    ccall((:cusolverSpCcsrqrZeroPivot, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, csrqrInfo_t, Cfloat, Ptr{Cint}), handle, info, tol, position)
end

@checked function cusolverSpZcsrqrZeroPivot(handle, info, tol, position)
    ccall((:cusolverSpZcsrqrZeroPivot, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, csrqrInfo_t, Cdouble, Ptr{Cint}), handle, info, tol, position)
end

@checked function cusolverSpScsrqrSolve(handle, m, n, b, x, info, pBuffer)
    ccall((:cusolverSpScsrqrSolve, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, csrqrInfo_t, Ptr{Cvoid}), handle, m, n, b, x, info, pBuffer)
end

@checked function cusolverSpDcsrqrSolve(handle, m, n, b, x, info, pBuffer)
    ccall((:cusolverSpDcsrqrSolve, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Ptr{Cdouble}, Ptr{Cdouble}, csrqrInfo_t, Ptr{Cvoid}), handle, m, n, b, x, info, pBuffer)
end

@checked function cusolverSpCcsrqrSolve(handle, m, n, b, x, info, pBuffer)
    ccall((:cusolverSpCcsrqrSolve, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Ptr{cuComplex}, Ptr{cuComplex}, csrqrInfo_t, Ptr{Cvoid}), handle, m, n, b, x, info, pBuffer)
end

@checked function cusolverSpZcsrqrSolve(handle, m, n, b, x, info, pBuffer)
    ccall((:cusolverSpZcsrqrSolve, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, csrqrInfo_t, Ptr{Cvoid}), handle, m, n, b, x, info, pBuffer)
end

@checked function cusolverSpCreateCsrcholInfoHost(info)
    ccall((:cusolverSpCreateCsrcholInfoHost, libcusolver()), cusolverStatus_t, (Ptr{csrcholInfoHost_t},), info)
end

@checked function cusolverSpDestroyCsrcholInfoHost(info)
    ccall((:cusolverSpDestroyCsrcholInfoHost, libcusolver()), cusolverStatus_t, (csrcholInfoHost_t,), info)
end

@checked function cusolverSpXcsrcholAnalysisHost(handle, n, nnzA, descrA, csrRowPtrA, csrColIndA, info)
    ccall((:cusolverSpXcsrcholAnalysisHost, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cint}, Ptr{Cint}, csrcholInfoHost_t), handle, n, nnzA, descrA, csrRowPtrA, csrColIndA, info)
end

@checked function cusolverSpScsrcholBufferInfoHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
    ccall((:cusolverSpScsrcholBufferInfoHost, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cfloat}, Ptr{Cint}, Ptr{Cint}, csrcholInfoHost_t, Ptr{Cint}, Ptr{Cint}), handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
end

@checked function cusolverSpDcsrcholBufferInfoHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
    ccall((:cusolverSpDcsrcholBufferInfoHost, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, csrcholInfoHost_t, Ptr{Cint}, Ptr{Cint}), handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
end

@checked function cusolverSpCcsrcholBufferInfoHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
    ccall((:cusolverSpCcsrcholBufferInfoHost, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{cuComplex}, Ptr{Cint}, Ptr{Cint}, csrcholInfoHost_t, Ptr{Cint}, Ptr{Cint}), handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
end

@checked function cusolverSpZcsrcholBufferInfoHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
    ccall((:cusolverSpZcsrcholBufferInfoHost, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, csrcholInfoHost_t, Ptr{Cint}, Ptr{Cint}), handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
end

@checked function cusolverSpScsrcholFactorHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pBuffer)
    ccall((:cusolverSpScsrcholFactorHost, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cfloat}, Ptr{Cint}, Ptr{Cint}, csrcholInfoHost_t, Ptr{Cvoid}), handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pBuffer)
end

@checked function cusolverSpDcsrcholFactorHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pBuffer)
    ccall((:cusolverSpDcsrcholFactorHost, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, csrcholInfoHost_t, Ptr{Cvoid}), handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pBuffer)
end

@checked function cusolverSpCcsrcholFactorHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pBuffer)
    ccall((:cusolverSpCcsrcholFactorHost, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{cuComplex}, Ptr{Cint}, Ptr{Cint}, csrcholInfoHost_t, Ptr{Cvoid}), handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pBuffer)
end

@checked function cusolverSpZcsrcholFactorHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pBuffer)
    ccall((:cusolverSpZcsrcholFactorHost, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, csrcholInfoHost_t, Ptr{Cvoid}), handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pBuffer)
end

@checked function cusolverSpScsrcholZeroPivotHost(handle, info, tol, position)
    ccall((:cusolverSpScsrcholZeroPivotHost, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, csrcholInfoHost_t, Cfloat, Ptr{Cint}), handle, info, tol, position)
end

@checked function cusolverSpDcsrcholZeroPivotHost(handle, info, tol, position)
    ccall((:cusolverSpDcsrcholZeroPivotHost, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, csrcholInfoHost_t, Cdouble, Ptr{Cint}), handle, info, tol, position)
end

@checked function cusolverSpCcsrcholZeroPivotHost(handle, info, tol, position)
    ccall((:cusolverSpCcsrcholZeroPivotHost, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, csrcholInfoHost_t, Cfloat, Ptr{Cint}), handle, info, tol, position)
end

@checked function cusolverSpZcsrcholZeroPivotHost(handle, info, tol, position)
    ccall((:cusolverSpZcsrcholZeroPivotHost, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, csrcholInfoHost_t, Cdouble, Ptr{Cint}), handle, info, tol, position)
end

@checked function cusolverSpScsrcholSolveHost(handle, n, b, x, info, pBuffer)
    ccall((:cusolverSpScsrcholSolveHost, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Ptr{Cfloat}, Ptr{Cfloat}, csrcholInfoHost_t, Ptr{Cvoid}), handle, n, b, x, info, pBuffer)
end

@checked function cusolverSpDcsrcholSolveHost(handle, n, b, x, info, pBuffer)
    ccall((:cusolverSpDcsrcholSolveHost, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Ptr{Cdouble}, Ptr{Cdouble}, csrcholInfoHost_t, Ptr{Cvoid}), handle, n, b, x, info, pBuffer)
end

@checked function cusolverSpCcsrcholSolveHost(handle, n, b, x, info, pBuffer)
    ccall((:cusolverSpCcsrcholSolveHost, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Ptr{cuComplex}, Ptr{cuComplex}, csrcholInfoHost_t, Ptr{Cvoid}), handle, n, b, x, info, pBuffer)
end

@checked function cusolverSpZcsrcholSolveHost(handle, n, b, x, info, pBuffer)
    ccall((:cusolverSpZcsrcholSolveHost, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, csrcholInfoHost_t, Ptr{Cvoid}), handle, n, b, x, info, pBuffer)
end

@checked function cusolverSpCreateCsrcholInfo(info)
    ccall((:cusolverSpCreateCsrcholInfo, libcusolver()), cusolverStatus_t, (Ptr{csrcholInfo_t},), info)
end

@checked function cusolverSpDestroyCsrcholInfo(info)
    ccall((:cusolverSpDestroyCsrcholInfo, libcusolver()), cusolverStatus_t, (csrcholInfo_t,), info)
end

@checked function cusolverSpXcsrcholAnalysis(handle, n, nnzA, descrA, csrRowPtrA, csrColIndA, info)
    ccall((:cusolverSpXcsrcholAnalysis, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cint}, Ptr{Cint}, csrcholInfo_t), handle, n, nnzA, descrA, csrRowPtrA, csrColIndA, info)
end

@checked function cusolverSpScsrcholBufferInfo(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
    ccall((:cusolverSpScsrcholBufferInfo, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cfloat}, Ptr{Cint}, Ptr{Cint}, csrcholInfo_t, Ptr{Cint}, Ptr{Cint}), handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
end

@checked function cusolverSpDcsrcholBufferInfo(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
    ccall((:cusolverSpDcsrcholBufferInfo, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, csrcholInfo_t, Ptr{Cint}, Ptr{Cint}), handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
end

@checked function cusolverSpCcsrcholBufferInfo(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
    ccall((:cusolverSpCcsrcholBufferInfo, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{cuComplex}, Ptr{Cint}, Ptr{Cint}, csrcholInfo_t, Ptr{Cint}, Ptr{Cint}), handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
end

@checked function cusolverSpZcsrcholBufferInfo(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
    ccall((:cusolverSpZcsrcholBufferInfo, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, csrcholInfo_t, Ptr{Cint}, Ptr{Cint}), handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
end

@checked function cusolverSpScsrcholFactor(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pBuffer)
    ccall((:cusolverSpScsrcholFactor, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cfloat}, Ptr{Cint}, Ptr{Cint}, csrcholInfo_t, Ptr{Cvoid}), handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pBuffer)
end

@checked function cusolverSpDcsrcholFactor(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pBuffer)
    ccall((:cusolverSpDcsrcholFactor, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, csrcholInfo_t, Ptr{Cvoid}), handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pBuffer)
end

@checked function cusolverSpCcsrcholFactor(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pBuffer)
    ccall((:cusolverSpCcsrcholFactor, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{cuComplex}, Ptr{Cint}, Ptr{Cint}, csrcholInfo_t, Ptr{Cvoid}), handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pBuffer)
end

@checked function cusolverSpZcsrcholFactor(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pBuffer)
    ccall((:cusolverSpZcsrcholFactor, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, csrcholInfo_t, Ptr{Cvoid}), handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pBuffer)
end

@checked function cusolverSpScsrcholZeroPivot(handle, info, tol, position)
    ccall((:cusolverSpScsrcholZeroPivot, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, csrcholInfo_t, Cfloat, Ptr{Cint}), handle, info, tol, position)
end

@checked function cusolverSpDcsrcholZeroPivot(handle, info, tol, position)
    ccall((:cusolverSpDcsrcholZeroPivot, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, csrcholInfo_t, Cdouble, Ptr{Cint}), handle, info, tol, position)
end

@checked function cusolverSpCcsrcholZeroPivot(handle, info, tol, position)
    ccall((:cusolverSpCcsrcholZeroPivot, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, csrcholInfo_t, Cfloat, Ptr{Cint}), handle, info, tol, position)
end

@checked function cusolverSpZcsrcholZeroPivot(handle, info, tol, position)
    ccall((:cusolverSpZcsrcholZeroPivot, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, csrcholInfo_t, Cdouble, Ptr{Cint}), handle, info, tol, position)
end

@checked function cusolverSpScsrcholSolve(handle, n, b, x, info, pBuffer)
    ccall((:cusolverSpScsrcholSolve, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Ptr{Cfloat}, Ptr{Cfloat}, csrcholInfo_t, Ptr{Cvoid}), handle, n, b, x, info, pBuffer)
end

@checked function cusolverSpDcsrcholSolve(handle, n, b, x, info, pBuffer)
    ccall((:cusolverSpDcsrcholSolve, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Ptr{Cdouble}, Ptr{Cdouble}, csrcholInfo_t, Ptr{Cvoid}), handle, n, b, x, info, pBuffer)
end

@checked function cusolverSpCcsrcholSolve(handle, n, b, x, info, pBuffer)
    ccall((:cusolverSpCcsrcholSolve, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Ptr{cuComplex}, Ptr{cuComplex}, csrcholInfo_t, Ptr{Cvoid}), handle, n, b, x, info, pBuffer)
end

@checked function cusolverSpZcsrcholSolve(handle, n, b, x, info, pBuffer)
    ccall((:cusolverSpZcsrcholSolve, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, csrcholInfo_t, Ptr{Cvoid}), handle, n, b, x, info, pBuffer)
end

@checked function cusolverSpScsrcholDiag(handle, info, diag)
    ccall((:cusolverSpScsrcholDiag, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, csrcholInfo_t, Ptr{Cfloat}), handle, info, diag)
end

@checked function cusolverSpDcsrcholDiag(handle, info, diag)
    ccall((:cusolverSpDcsrcholDiag, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, csrcholInfo_t, Ptr{Cdouble}), handle, info, diag)
end

@checked function cusolverSpCcsrcholDiag(handle, info, diag)
    ccall((:cusolverSpCcsrcholDiag, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, csrcholInfo_t, Ptr{Cfloat}), handle, info, diag)
end

@checked function cusolverSpZcsrcholDiag(handle, info, diag)
    ccall((:cusolverSpZcsrcholDiag, libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, csrcholInfo_t, Ptr{Cdouble}), handle, info, diag)
end

@checked function cusparseSpSV_createDescr(descr)
    CUDA.initialize_context()
    ccall((:cusparseSpSV_createDescr, libcusparse), cusparseStatus_t, (Ptr{cusparseSpSVDescr_t},), descr)
end

@checked function cusparseSpSV_destroyDescr(descr)
    CUDA.initialize_context()
    ccall((:cusparseSpSV_destroyDescr, libcusparse), cusparseStatus_t, (cusparseSpSVDescr_t,), descr)
end

@checked function cusparseSpSV_bufferSize(handle, opA, alpha, matA, vecX, vecY, computeType, alg, spsvDescr, bufferSize)
    CUDA.initialize_context()
    ccall((:cusparseSpSV_bufferSize, libcusparse), cusparseStatus_t, (cusparseHandle_t, cusparseOperation_t, Ptr{Cvoid}, cusparseSpMatDescr_t, cusparseDnVecDescr_t, cusparseDnVecDescr_t, Cint, cusparseSpSVAlg_t, cusparseSpSVDescr_t, Ptr{Csize_t}), handle, opA, alpha, matA, vecX, vecY, computeType, alg, spsvDescr, bufferSize)
end

@checked function cusparseSpSV_analysis(handle, opA, alpha, matA, vecX, vecY, computeType, alg, spsvDescr, externalBuffer)
    CUDA.initialize_context()
    ccall((:cusparseSpSV_analysis, libcusparse), cusparseStatus_t, (cusparseHandle_t, cusparseOperation_t, Ptr{Cvoid}, cusparseSpMatDescr_t, cusparseDnVecDescr_t, cusparseDnVecDescr_t, Cint, cusparseSpSVAlg_t, cusparseSpSVDescr_t, Ptr{Cvoid}), handle, opA, alpha, matA, vecX, vecY, computeType, alg, spsvDescr, externalBuffer)
end

@checked function cusparseSpSV_solve(handle, opA, alpha, matA, vecX, vecY, computeType, alg, spsvDescr)
    CUDA.initialize_context()
    ccall((:cusparseSpSV_solve, libcusparse), cusparseStatus_t, (cusparseHandle_t, cusparseOperation_t, Ptr{Cvoid}, cusparseSpMatDescr_t, cusparseDnVecDescr_t, cusparseDnVecDescr_t, Cint, cusparseSpSVAlg_t, cusparseSpSVDescr_t), handle, opA, alpha, matA, vecX, vecY, computeType, alg, spsvDescr)
end

@checked function cusparseSpSM_createDescr(descr)
    CUDA.initialize_context()
    ccall((:cusparseSpSM_createDescr, libcusparse), cusparseStatus_t, (Ptr{cusparseSpSMDescr_t},), descr)
end

@checked function cusparseSpSM_destroyDescr(descr)
    CUDA.initialize_context()
    ccall((:cusparseSpSM_destroyDescr, libcusparse), cusparseStatus_t, (cusparseSpSMDescr_t,), descr)
end

@checked function cusparseSpSM_bufferSize(handle, opA, opB, alpha, matA, matB, matC, computeType, alg, spsmDescr, bufferSize)
    CUDA.initialize_context()
    ccall((:cusparseSpSM_bufferSize, libcusparse), cusparseStatus_t, (cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, Ptr{Cvoid}, cusparseSpMatDescr_t, cusparseDnMatDescr_t, cusparseDnMatDescr_t, Cint, cusparseSpSMAlg_t, cusparseSpSMDescr_t, Ptr{Csize_t}), handle, opA, opB, alpha, matA, matB, matC, computeType, alg, spsmDescr, bufferSize)
end

@checked function cusparseSpSM_analysis(handle, opA, opB, alpha, matA, matB, matC, computeType, alg, spsmDescr, externalBuffer)
    CUDA.initialize_context()
    ccall((:cusparseSpSM_analysis, libcusparse), cusparseStatus_t, (cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, Ptr{Cvoid}, cusparseSpMatDescr_t, cusparseDnMatDescr_t, cusparseDnMatDescr_t, Cint, cusparseSpSMAlg_t, cusparseSpSMDescr_t, Ptr{Cvoid}), handle, opA, opB, alpha, matA, matB, matC, computeType, alg, spsmDescr, externalBuffer)
end

@checked function cusparseSpSM_solve(handle, opA, opB, alpha, matA, matB, matC, computeType, alg, spsmDescr)
    CUDA.initialize_context()
    ccall((:cusparseSpSM_solve, libcusparse), cusparseStatus_t, (cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, Ptr{Cvoid}, cusparseSpMatDescr_t, cusparseDnMatDescr_t, cusparseDnMatDescr_t, Cint, cusparseSpSMAlg_t, cusparseSpSMDescr_t), handle, opA, opB, alpha, matA, matB, matC, computeType, alg, spsmDescr)
end

@checked function cusparseSpMatGetAttribute(spMatDescr, attribute, data, dataSize)
    CUDA.initialize_context()
    ccall((:cusparseSpMatGetAttribute, libcusparse), cusparseStatus_t, (cusparseSpMatDescr_t, cusparseSpMatAttribute_t, Ptr{Cvoid}, Csize_t), spMatDescr, attribute, data, dataSize)
end

@checked function cusparseSpMatSetAttribute(spMatDescr, attribute, data, dataSize)
    CUDA.initialize_context()
    ccall((:cusparseSpMatSetAttribute, libcusparse), cusparseStatus_t, (cusparseSpMatDescr_t, cusparseSpMatAttribute_t, Ptr{Cvoid}, Csize_t), spMatDescr, attribute, data, dataSize)
end

