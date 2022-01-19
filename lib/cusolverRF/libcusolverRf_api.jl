# Julia wrapper for header: cusolverRf.h
# Automatically generated using Clang.jl

import CUDA: cuComplex, cuDoubleComplex
import CUDA.CUSOLVER: cusolverStatus_t
import CUDA.CUSPARSE: cusparseMatDescr_t
import CUDA: @check
import CUDA.APIUtils: @checked

@checked function cusolverRfCreate(handle)
    ccall((:cusolverRfCreate, CUSOLVER.libcusolver()), cusolverStatus_t, (Ptr{cusolverRfHandle_t},), handle)
end

@checked function cusolverRfDestroy(handle)
    ccall((:cusolverRfDestroy, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverRfHandle_t,), handle)
end

@checked function cusolverRfGetMatrixFormat(handle, format, diag)
    ccall((:cusolverRfGetMatrixFormat, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverRfHandle_t, Ptr{cusolverRfMatrixFormat_t}, Ptr{cusolverRfUnitDiagonal_t}), handle, format, diag)
end

@checked function cusolverRfSetMatrixFormat(handle, format, diag)
    ccall((:cusolverRfSetMatrixFormat, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverRfHandle_t, cusolverRfMatrixFormat_t, cusolverRfUnitDiagonal_t), handle, format, diag)
end

@checked function cusolverRfSetNumericProperties(handle, zero, boost)
    ccall((:cusolverRfSetNumericProperties, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverRfHandle_t, Cdouble, Cdouble), handle, zero, boost)
end

@checked function cusolverRfGetNumericProperties(handle, zero, boost)
    ccall((:cusolverRfGetNumericProperties, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverRfHandle_t, Ptr{Cdouble}, Ptr{Cdouble}), handle, zero, boost)
end

@checked function cusolverRfGetNumericBoostReport(handle, report)
    ccall((:cusolverRfGetNumericBoostReport, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverRfHandle_t, Ptr{cusolverRfNumericBoostReport_t}), handle, report)
end

@checked function cusolverRfSetAlgs(handle, factAlg, solveAlg)
    ccall((:cusolverRfSetAlgs, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverRfHandle_t, cusolverRfFactorization_t, cusolverRfTriangularSolve_t), handle, factAlg, solveAlg)
end

@checked function cusolverRfGetAlgs(handle, factAlg, solveAlg)
    ccall((:cusolverRfGetAlgs, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverRfHandle_t, Ptr{cusolverRfFactorization_t}, Ptr{cusolverRfTriangularSolve_t}), handle, factAlg, solveAlg)
end

@checked function cusolverRfGetResetValuesFastMode(handle, fastMode)
    ccall((:cusolverRfGetResetValuesFastMode, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverRfHandle_t, Ptr{cusolverRfResetValuesFastMode_t}), handle, fastMode)
end

@checked function cusolverRfSetResetValuesFastMode(handle, fastMode)
    ccall((:cusolverRfSetResetValuesFastMode, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverRfHandle_t, cusolverRfResetValuesFastMode_t), handle, fastMode)
end

@checked function cusolverRfSetupHost(n, nnzA, h_csrRowPtrA, h_csrColIndA, h_csrValA, nnzL, h_csrRowPtrL, h_csrColIndL, h_csrValL, nnzU, h_csrRowPtrU, h_csrColIndU, h_csrValU, h_P, h_Q, handle)
    ccall((:cusolverRfSetupHost, CUSOLVER.libcusolver()), cusolverStatus_t, (Cint, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, cusolverRfHandle_t), n, nnzA, h_csrRowPtrA, h_csrColIndA, h_csrValA, nnzL, h_csrRowPtrL, h_csrColIndL, h_csrValL, nnzU, h_csrRowPtrU, h_csrColIndU, h_csrValU, h_P, h_Q, handle)
end

@checked function cusolverRfSetupDevice(n, nnzA, csrRowPtrA, csrColIndA, csrValA, nnzL, csrRowPtrL, csrColIndL, csrValL, nnzU, csrRowPtrU, csrColIndU, csrValU, P, Q, handle)
    ccall((:cusolverRfSetupDevice, CUSOLVER.libcusolver()), cusolverStatus_t, (Cint, Cint, CuPtr{Cint}, CuPtr{Cint}, CuPtr{Cdouble}, Cint, CuPtr{Cint}, CuPtr{Cint}, CuPtr{Cdouble}, Cint, CuPtr{Cint}, CuPtr{Cint}, CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint}, cusolverRfHandle_t), n, nnzA, csrRowPtrA, csrColIndA, csrValA, nnzL, csrRowPtrL, csrColIndL, csrValL, nnzU, csrRowPtrU, csrColIndU, csrValU, P, Q, handle)
end

@checked function cusolverRfResetValues(n, nnzA, csrRowPtrA, csrColIndA, csrValA, P, Q, handle)
    ccall((:cusolverRfResetValues, CUSOLVER.libcusolver()), cusolverStatus_t, (Cint, Cint, CuPtr{Cint}, CuPtr{Cint}, CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint}, cusolverRfHandle_t), n, nnzA, csrRowPtrA, csrColIndA, csrValA, P, Q, handle)
end

@checked function cusolverRfAnalyze(handle)
    ccall((:cusolverRfAnalyze, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverRfHandle_t,), handle)
end

@checked function cusolverRfRefactor(handle)
    ccall((:cusolverRfRefactor, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverRfHandle_t,), handle)
end

@checked function cusolverRfAccessBundledFactorsDevice(handle, nnzM, Mp, Mi, Mx)
    ccall((:cusolverRfAccessBundledFactorsDevice, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverRfHandle_t, Ptr{Cint}, Ptr{Ptr{Cint}}, Ptr{Ptr{Cint}}, Ptr{Ptr{Cdouble}}), handle, nnzM, Mp, Mi, Mx)
end

@checked function cusolverRfExtractBundledFactorsHost(handle, h_nnzM, h_Mp, h_Mi, h_Mx)
    ccall((:cusolverRfExtractBundledFactorsHost, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverRfHandle_t, Ptr{Cint}, Ptr{Ptr{Cint}}, Ptr{Ptr{Cint}}, Ptr{Ptr{Cdouble}}), handle, h_nnzM, h_Mp, h_Mi, h_Mx)
end

@checked function cusolverRfExtractSplitFactorsHost(handle, h_nnzL, h_csrRowPtrL, h_csrColIndL, h_csrValL, h_nnzU, h_csrRowPtrU, h_csrColIndU, h_csrValU)
    ccall((:cusolverRfExtractSplitFactorsHost, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverRfHandle_t, Ptr{Cint}, Ptr{Ptr{Cint}}, Ptr{Ptr{Cint}}, Ptr{Ptr{Cdouble}}, Ptr{Cint}, Ptr{Ptr{Cint}}, Ptr{Ptr{Cint}}, Ptr{Ptr{Cdouble}}), handle, h_nnzL, h_csrRowPtrL, h_csrColIndL, h_csrValL, h_nnzU, h_csrRowPtrU, h_csrColIndU, h_csrValU)
end

function cusolverRfSolve(handle, P, Q, nrhs, Temp, ldt, XF, ldxf)
    ccall((:cusolverRfSolve, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverRfHandle_t, CuPtr{Cint}, CuPtr{Cint}, Cint, CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint), handle, P, Q, nrhs, Temp, ldt, XF, ldxf)
end

@checked function cusolverRfBatchSetupHost(batchSize, n, nnzA, h_csrRowPtrA, h_csrColIndA, h_csrValA_array, nnzL, h_csrRowPtrL, h_csrColIndL, h_csrValL, nnzU, h_csrRowPtrU, h_csrColIndU, h_csrValU, h_P, h_Q, handle)
    ccall((:cusolverRfBatchSetupHost, CUSOLVER.libcusolver()), cusolverStatus_t, (Cint, Cint, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Ptr{Cdouble}}, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, cusolverRfHandle_t), batchSize, n, nnzA, h_csrRowPtrA, h_csrColIndA, h_csrValA_array, nnzL, h_csrRowPtrL, h_csrColIndL, h_csrValL, nnzU, h_csrRowPtrU, h_csrColIndU, h_csrValU, h_P, h_Q, handle)
end

@checked function cusolverRfBatchResetValues(batchSize, n, nnzA, csrRowPtrA, csrColIndA, csrValA_array, P, Q, handle)
    ccall((:cusolverRfBatchResetValues, CUSOLVER.libcusolver()), cusolverStatus_t, (Cint, Cint, Cint, CuPtr{Cint}, CuPtr{Cint}, CuPtr{Ptr{Cdouble}}, CuPtr{Cint}, CuPtr{Cint}, cusolverRfHandle_t), batchSize, n, nnzA, csrRowPtrA, csrColIndA, csrValA_array, P, Q, handle)
end

@checked function cusolverRfBatchAnalyze(handle)
    ccall((:cusolverRfBatchAnalyze, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverRfHandle_t,), handle)
end

@checked function cusolverRfBatchRefactor(handle)
    ccall((:cusolverRfBatchRefactor, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverRfHandle_t,), handle)
end

@checked function cusolverRfBatchSolve(handle, P, Q, nrhs, Temp, ldt, XF_array, ldxf)
    ccall((:cusolverRfBatchSolve, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverRfHandle_t, CuPtr{Cint}, CuPtr{Cint}, Cint, CuPtr{Cdouble}, Cint, CuPtr{Ptr{Cdouble}}, Cint), handle, P, Q, nrhs, Temp, ldt, XF_array, ldxf)
end

@checked function cusolverRfBatchZeroPivot(handle, position)
    ccall((:cusolverRfBatchZeroPivot, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverRfHandle_t, Ptr{Cint}), handle, position)
end

# Julia wrapper for header: cusolverSp_LOWLEVEL_PREVIEW.h
# Automatically generated using Clang.jl


@checked function cusolverSpCreateCsrluInfoHost(info)
    ccall((:cusolverSpCreateCsrluInfoHost, CUSOLVER.libcusolver()), cusolverStatus_t, (Ptr{csrluInfoHost_t},), info)
end

@checked function cusolverSpDestroyCsrluInfoHost(info)
    ccall((:cusolverSpDestroyCsrluInfoHost, CUSOLVER.libcusolver()), cusolverStatus_t, (csrluInfoHost_t,), info)
end

@checked function cusolverSpXcsrluAnalysisHost(handle, n, nnzA, descrA, csrRowPtrA, csrColIndA, info)
    ccall((:cusolverSpXcsrluAnalysisHost, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cint}, Ptr{Cint}, csrluInfoHost_t), handle, n, nnzA, descrA, csrRowPtrA, csrColIndA, info)
end

@checked function cusolverSpScsrluBufferInfoHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
    ccall((:cusolverSpScsrluBufferInfoHost, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cfloat}, Ptr{Cint}, Ptr{Cint}, csrluInfoHost_t, Ptr{Cint}, Ptr{Cint}), handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
end

@checked function cusolverSpDcsrluBufferInfoHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
    ccall((:cusolverSpDcsrluBufferInfoHost, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, csrluInfoHost_t, Ptr{Cint}, Ptr{Cint}), handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
end

@checked function cusolverSpCcsrluBufferInfoHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
    ccall((:cusolverSpCcsrluBufferInfoHost, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{cuComplex}, Ptr{Cint}, Ptr{Cint}, csrluInfoHost_t, Ptr{Cint}, Ptr{Cint}), handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
end

@checked function cusolverSpZcsrluBufferInfoHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
    ccall((:cusolverSpZcsrluBufferInfoHost, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, csrluInfoHost_t, Ptr{Cint}, Ptr{Cint}), handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
end

@checked function cusolverSpScsrluFactorHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pivot_threshold, pBuffer)
    ccall((:cusolverSpScsrluFactorHost, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cfloat}, Ptr{Cint}, Ptr{Cint}, csrluInfoHost_t, Cfloat, Ptr{Cvoid}), handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pivot_threshold, pBuffer)
end

@checked function cusolverSpDcsrluFactorHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pivot_threshold, pBuffer)
    ccall((:cusolverSpDcsrluFactorHost, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, csrluInfoHost_t, Cdouble, Ptr{Cvoid}), handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pivot_threshold, pBuffer)
end

@checked function cusolverSpCcsrluFactorHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pivot_threshold, pBuffer)
    ccall((:cusolverSpCcsrluFactorHost, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{cuComplex}, Ptr{Cint}, Ptr{Cint}, csrluInfoHost_t, Cfloat, Ptr{Cvoid}), handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pivot_threshold, pBuffer)
end

@checked function cusolverSpZcsrluFactorHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pivot_threshold, pBuffer)
    ccall((:cusolverSpZcsrluFactorHost, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, csrluInfoHost_t, Cdouble, Ptr{Cvoid}), handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pivot_threshold, pBuffer)
end

@checked function cusolverSpScsrluZeroPivotHost(handle, info, tol, position)
    ccall((:cusolverSpScsrluZeroPivotHost, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, csrluInfoHost_t, Cfloat, Ptr{Cint}), handle, info, tol, position)
end

@checked function cusolverSpDcsrluZeroPivotHost(handle, info, tol, position)
    ccall((:cusolverSpDcsrluZeroPivotHost, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, csrluInfoHost_t, Cdouble, Ptr{Cint}), handle, info, tol, position)
end

@checked function cusolverSpCcsrluZeroPivotHost(handle, info, tol, position)
    ccall((:cusolverSpCcsrluZeroPivotHost, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, csrluInfoHost_t, Cfloat, Ptr{Cint}), handle, info, tol, position)
end

@checked function cusolverSpZcsrluZeroPivotHost(handle, info, tol, position)
    ccall((:cusolverSpZcsrluZeroPivotHost, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, csrluInfoHost_t, Cdouble, Ptr{Cint}), handle, info, tol, position)
end

@checked function cusolverSpScsrluSolveHost(handle, n, b, x, info, pBuffer)
    ccall((:cusolverSpScsrluSolveHost, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Ptr{Cfloat}, Ptr{Cfloat}, csrluInfoHost_t, Ptr{Cvoid}), handle, n, b, x, info, pBuffer)
end

@checked function cusolverSpDcsrluSolveHost(handle, n, b, x, info, pBuffer)
    ccall((:cusolverSpDcsrluSolveHost, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Ptr{Cdouble}, Ptr{Cdouble}, csrluInfoHost_t, Ptr{Cvoid}), handle, n, b, x, info, pBuffer)
end

@checked function cusolverSpCcsrluSolveHost(handle, n, b, x, info, pBuffer)
    ccall((:cusolverSpCcsrluSolveHost, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Ptr{cuComplex}, Ptr{cuComplex}, csrluInfoHost_t, Ptr{Cvoid}), handle, n, b, x, info, pBuffer)
end

@checked function cusolverSpZcsrluSolveHost(handle, n, b, x, info, pBuffer)
    ccall((:cusolverSpZcsrluSolveHost, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, csrluInfoHost_t, Ptr{Cvoid}), handle, n, b, x, info, pBuffer)
end

@checked function cusolverSpXcsrluNnzHost(handle, nnzLRef, nnzURef, info)
    ccall((:cusolverSpXcsrluNnzHost, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Ptr{Cint}, Ptr{Cint}, csrluInfoHost_t), handle, nnzLRef, nnzURef, info)
end

@checked function cusolverSpScsrluExtractHost(handle, P, Q, descrL, csrValL, csrRowPtrL, csrColIndL, descrU, csrValU, csrRowPtrU, csrColIndU, info, pBuffer)
    ccall((:cusolverSpScsrluExtractHost, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Ptr{Cint}, Ptr{Cint}, cusparseMatDescr_t, Ptr{Cfloat}, Ptr{Cint}, Ptr{Cint}, cusparseMatDescr_t, Ptr{Cfloat}, Ptr{Cint}, Ptr{Cint}, csrluInfoHost_t, Ptr{Cvoid}), handle, P, Q, descrL, csrValL, csrRowPtrL, csrColIndL, descrU, csrValU, csrRowPtrU, csrColIndU, info, pBuffer)
end

@checked function cusolverSpDcsrluExtractHost(handle, P, Q, descrL, csrValL, csrRowPtrL, csrColIndL, descrU, csrValU, csrRowPtrU, csrColIndU, info, pBuffer)
    ccall((:cusolverSpDcsrluExtractHost, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Ptr{Cint}, Ptr{Cint}, cusparseMatDescr_t, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, cusparseMatDescr_t, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, csrluInfoHost_t, Ptr{Cvoid}), handle, P, Q, descrL, csrValL, csrRowPtrL, csrColIndL, descrU, csrValU, csrRowPtrU, csrColIndU, info, pBuffer)
end

@checked function cusolverSpCcsrluExtractHost(handle, P, Q, descrL, csrValL, csrRowPtrL, csrColIndL, descrU, csrValU, csrRowPtrU, csrColIndU, info, pBuffer)
    ccall((:cusolverSpCcsrluExtractHost, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Ptr{Cint}, Ptr{Cint}, cusparseMatDescr_t, Ptr{cuComplex}, Ptr{Cint}, Ptr{Cint}, cusparseMatDescr_t, Ptr{cuComplex}, Ptr{Cint}, Ptr{Cint}, csrluInfoHost_t, Ptr{Cvoid}), handle, P, Q, descrL, csrValL, csrRowPtrL, csrColIndL, descrU, csrValU, csrRowPtrU, csrColIndU, info, pBuffer)
end

@checked function cusolverSpZcsrluExtractHost(handle, P, Q, descrL, csrValL, csrRowPtrL, csrColIndL, descrU, csrValU, csrRowPtrU, csrColIndU, info, pBuffer)
    ccall((:cusolverSpZcsrluExtractHost, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Ptr{Cint}, Ptr{Cint}, cusparseMatDescr_t, Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, cusparseMatDescr_t, Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, csrluInfoHost_t, Ptr{Cvoid}), handle, P, Q, descrL, csrValL, csrRowPtrL, csrColIndL, descrU, csrValU, csrRowPtrU, csrColIndU, info, pBuffer)
end

@checked function cusolverSpCreateCsrqrInfoHost(info)
    ccall((:cusolverSpCreateCsrqrInfoHost, CUSOLVER.libcusolver()), cusolverStatus_t, (Ptr{csrqrInfoHost_t},), info)
end

@checked function cusolverSpDestroyCsrqrInfoHost(info)
    ccall((:cusolverSpDestroyCsrqrInfoHost, CUSOLVER.libcusolver()), cusolverStatus_t, (csrqrInfoHost_t,), info)
end

@checked function cusolverSpXcsrqrAnalysisHost(handle, m, n, nnzA, descrA, csrRowPtrA, csrColIndA, info)
    ccall((:cusolverSpXcsrqrAnalysisHost, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, Ptr{Cint}, Ptr{Cint}, csrqrInfoHost_t), handle, m, n, nnzA, descrA, csrRowPtrA, csrColIndA, info)
end

@checked function cusolverSpScsrqrBufferInfoHost(handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
    ccall((:cusolverSpScsrqrBufferInfoHost, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, Ptr{Cfloat}, Ptr{Cint}, Ptr{Cint}, csrqrInfoHost_t, Ptr{Cint}, Ptr{Cint}), handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
end

@checked function cusolverSpDcsrqrBufferInfoHost(handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
    ccall((:cusolverSpDcsrqrBufferInfoHost, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, csrqrInfoHost_t, Ptr{Cint}, Ptr{Cint}), handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
end

@checked function cusolverSpCcsrqrBufferInfoHost(handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
    ccall((:cusolverSpCcsrqrBufferInfoHost, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, Ptr{cuComplex}, Ptr{Cint}, Ptr{Cint}, csrqrInfoHost_t, Ptr{Cint}, Ptr{Cint}), handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
end

@checked function cusolverSpZcsrqrBufferInfoHost(handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
    ccall((:cusolverSpZcsrqrBufferInfoHost, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, csrqrInfoHost_t, Ptr{Cint}, Ptr{Cint}), handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
end

@checked function cusolverSpScsrqrSetupHost(handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, mu, info)
    ccall((:cusolverSpScsrqrSetupHost, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, Ptr{Cfloat}, Ptr{Cint}, Ptr{Cint}, Cfloat, csrqrInfoHost_t), handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, mu, info)
end

@checked function cusolverSpDcsrqrSetupHost(handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, mu, info)
    ccall((:cusolverSpDcsrqrSetupHost, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, Cdouble, csrqrInfoHost_t), handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, mu, info)
end

@checked function cusolverSpCcsrqrSetupHost(handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, mu, info)
    ccall((:cusolverSpCcsrqrSetupHost, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, Ptr{cuComplex}, Ptr{Cint}, Ptr{Cint}, cuComplex, csrqrInfoHost_t), handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, mu, info)
end

@checked function cusolverSpZcsrqrSetupHost(handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, mu, info)
    ccall((:cusolverSpZcsrqrSetupHost, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, cuDoubleComplex, csrqrInfoHost_t), handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, mu, info)
end

@checked function cusolverSpScsrqrFactorHost(handle, m, n, nnzA, b, x, info, pBuffer)
    ccall((:cusolverSpScsrqrFactorHost, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, csrqrInfoHost_t, Ptr{Cvoid}), handle, m, n, nnzA, b, x, info, pBuffer)
end

@checked function cusolverSpDcsrqrFactorHost(handle, m, n, nnzA, b, x, info, pBuffer)
    ccall((:cusolverSpDcsrqrFactorHost, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Cint, Ptr{Cdouble}, Ptr{Cdouble}, csrqrInfoHost_t, Ptr{Cvoid}), handle, m, n, nnzA, b, x, info, pBuffer)
end

@checked function cusolverSpCcsrqrFactorHost(handle, m, n, nnzA, b, x, info, pBuffer)
    ccall((:cusolverSpCcsrqrFactorHost, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Cint, Ptr{cuComplex}, Ptr{cuComplex}, csrqrInfoHost_t, Ptr{Cvoid}), handle, m, n, nnzA, b, x, info, pBuffer)
end

@checked function cusolverSpZcsrqrFactorHost(handle, m, n, nnzA, b, x, info, pBuffer)
    ccall((:cusolverSpZcsrqrFactorHost, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, csrqrInfoHost_t, Ptr{Cvoid}), handle, m, n, nnzA, b, x, info, pBuffer)
end

@checked function cusolverSpScsrqrZeroPivotHost(handle, info, tol, position)
    ccall((:cusolverSpScsrqrZeroPivotHost, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, csrqrInfoHost_t, Cfloat, Ptr{Cint}), handle, info, tol, position)
end

@checked function cusolverSpDcsrqrZeroPivotHost(handle, info, tol, position)
    ccall((:cusolverSpDcsrqrZeroPivotHost, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, csrqrInfoHost_t, Cdouble, Ptr{Cint}), handle, info, tol, position)
end

@checked function cusolverSpCcsrqrZeroPivotHost(handle, info, tol, position)
    ccall((:cusolverSpCcsrqrZeroPivotHost, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, csrqrInfoHost_t, Cfloat, Ptr{Cint}), handle, info, tol, position)
end

@checked function cusolverSpZcsrqrZeroPivotHost(handle, info, tol, position)
    ccall((:cusolverSpZcsrqrZeroPivotHost, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, csrqrInfoHost_t, Cdouble, Ptr{Cint}), handle, info, tol, position)
end

@checked function cusolverSpScsrqrSolveHost(handle, m, n, b, x, info, pBuffer)
    ccall((:cusolverSpScsrqrSolveHost, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, csrqrInfoHost_t, Ptr{Cvoid}), handle, m, n, b, x, info, pBuffer)
end

@checked function cusolverSpDcsrqrSolveHost(handle, m, n, b, x, info, pBuffer)
    ccall((:cusolverSpDcsrqrSolveHost, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Ptr{Cdouble}, Ptr{Cdouble}, csrqrInfoHost_t, Ptr{Cvoid}), handle, m, n, b, x, info, pBuffer)
end

@checked function cusolverSpCcsrqrSolveHost(handle, m, n, b, x, info, pBuffer)
    ccall((:cusolverSpCcsrqrSolveHost, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Ptr{cuComplex}, Ptr{cuComplex}, csrqrInfoHost_t, Ptr{Cvoid}), handle, m, n, b, x, info, pBuffer)
end

@checked function cusolverSpZcsrqrSolveHost(handle, m, n, b, x, info, pBuffer)
    ccall((:cusolverSpZcsrqrSolveHost, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, csrqrInfoHost_t, Ptr{Cvoid}), handle, m, n, b, x, info, pBuffer)
end

@checked function cusolverSpXcsrqrAnalysis(handle, m, n, nnzA, descrA, csrRowPtrA, csrColIndA, info)
    ccall((:cusolverSpXcsrqrAnalysis, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, Ptr{Cint}, Ptr{Cint}, csrqrInfo_t), handle, m, n, nnzA, descrA, csrRowPtrA, csrColIndA, info)
end

@checked function cusolverSpScsrqrBufferInfo(handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
    ccall((:cusolverSpScsrqrBufferInfo, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, Ptr{Cfloat}, Ptr{Cint}, Ptr{Cint}, csrqrInfo_t, Ptr{Cint}, Ptr{Cint}), handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
end

@checked function cusolverSpDcsrqrBufferInfo(handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
    ccall((:cusolverSpDcsrqrBufferInfo, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, csrqrInfo_t, Ptr{Cint}, Ptr{Cint}), handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
end

@checked function cusolverSpCcsrqrBufferInfo(handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
    ccall((:cusolverSpCcsrqrBufferInfo, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, Ptr{cuComplex}, Ptr{Cint}, Ptr{Cint}, csrqrInfo_t, Ptr{Cint}, Ptr{Cint}), handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
end

@checked function cusolverSpZcsrqrBufferInfo(handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
    ccall((:cusolverSpZcsrqrBufferInfo, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, csrqrInfo_t, Ptr{Cint}, Ptr{Cint}), handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
end

@checked function cusolverSpScsrqrSetup(handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, mu, info)
    ccall((:cusolverSpScsrqrSetup, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, Ptr{Cfloat}, Ptr{Cint}, Ptr{Cint}, Cfloat, csrqrInfo_t), handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, mu, info)
end

@checked function cusolverSpDcsrqrSetup(handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, mu, info)
    ccall((:cusolverSpDcsrqrSetup, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, Cdouble, csrqrInfo_t), handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, mu, info)
end

@checked function cusolverSpCcsrqrSetup(handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, mu, info)
    ccall((:cusolverSpCcsrqrSetup, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, Ptr{cuComplex}, Ptr{Cint}, Ptr{Cint}, cuComplex, csrqrInfo_t), handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, mu, info)
end

@checked function cusolverSpZcsrqrSetup(handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, mu, info)
    ccall((:cusolverSpZcsrqrSetup, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, cuDoubleComplex, csrqrInfo_t), handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, mu, info)
end

@checked function cusolverSpScsrqrFactor(handle, m, n, nnzA, b, x, info, pBuffer)
    ccall((:cusolverSpScsrqrFactor, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, csrqrInfo_t, Ptr{Cvoid}), handle, m, n, nnzA, b, x, info, pBuffer)
end

@checked function cusolverSpDcsrqrFactor(handle, m, n, nnzA, b, x, info, pBuffer)
    ccall((:cusolverSpDcsrqrFactor, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Cint, Ptr{Cdouble}, Ptr{Cdouble}, csrqrInfo_t, Ptr{Cvoid}), handle, m, n, nnzA, b, x, info, pBuffer)
end

@checked function cusolverSpCcsrqrFactor(handle, m, n, nnzA, b, x, info, pBuffer)
    ccall((:cusolverSpCcsrqrFactor, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Cint, Ptr{cuComplex}, Ptr{cuComplex}, csrqrInfo_t, Ptr{Cvoid}), handle, m, n, nnzA, b, x, info, pBuffer)
end

@checked function cusolverSpZcsrqrFactor(handle, m, n, nnzA, b, x, info, pBuffer)
    ccall((:cusolverSpZcsrqrFactor, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, csrqrInfo_t, Ptr{Cvoid}), handle, m, n, nnzA, b, x, info, pBuffer)
end

@checked function cusolverSpScsrqrZeroPivot(handle, info, tol, position)
    ccall((:cusolverSpScsrqrZeroPivot, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, csrqrInfo_t, Cfloat, Ptr{Cint}), handle, info, tol, position)
end

@checked function cusolverSpDcsrqrZeroPivot(handle, info, tol, position)
    ccall((:cusolverSpDcsrqrZeroPivot, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, csrqrInfo_t, Cdouble, Ptr{Cint}), handle, info, tol, position)
end

@checked function cusolverSpCcsrqrZeroPivot(handle, info, tol, position)
    ccall((:cusolverSpCcsrqrZeroPivot, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, csrqrInfo_t, Cfloat, Ptr{Cint}), handle, info, tol, position)
end

@checked function cusolverSpZcsrqrZeroPivot(handle, info, tol, position)
    ccall((:cusolverSpZcsrqrZeroPivot, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, csrqrInfo_t, Cdouble, Ptr{Cint}), handle, info, tol, position)
end

@checked function cusolverSpScsrqrSolve(handle, m, n, b, x, info, pBuffer)
    ccall((:cusolverSpScsrqrSolve, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, csrqrInfo_t, Ptr{Cvoid}), handle, m, n, b, x, info, pBuffer)
end

@checked function cusolverSpDcsrqrSolve(handle, m, n, b, x, info, pBuffer)
    ccall((:cusolverSpDcsrqrSolve, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Ptr{Cdouble}, Ptr{Cdouble}, csrqrInfo_t, Ptr{Cvoid}), handle, m, n, b, x, info, pBuffer)
end

@checked function cusolverSpCcsrqrSolve(handle, m, n, b, x, info, pBuffer)
    ccall((:cusolverSpCcsrqrSolve, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Ptr{cuComplex}, Ptr{cuComplex}, csrqrInfo_t, Ptr{Cvoid}), handle, m, n, b, x, info, pBuffer)
end

@checked function cusolverSpZcsrqrSolve(handle, m, n, b, x, info, pBuffer)
    ccall((:cusolverSpZcsrqrSolve, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, csrqrInfo_t, Ptr{Cvoid}), handle, m, n, b, x, info, pBuffer)
end

@checked function cusolverSpCreateCsrcholInfoHost(info)
    ccall((:cusolverSpCreateCsrcholInfoHost, CUSOLVER.libcusolver()), cusolverStatus_t, (Ptr{csrcholInfoHost_t},), info)
end

@checked function cusolverSpDestroyCsrcholInfoHost(info)
    ccall((:cusolverSpDestroyCsrcholInfoHost, CUSOLVER.libcusolver()), cusolverStatus_t, (csrcholInfoHost_t,), info)
end

@checked function cusolverSpXcsrcholAnalysisHost(handle, n, nnzA, descrA, csrRowPtrA, csrColIndA, info)
    ccall((:cusolverSpXcsrcholAnalysisHost, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cint}, Ptr{Cint}, csrcholInfoHost_t), handle, n, nnzA, descrA, csrRowPtrA, csrColIndA, info)
end

@checked function cusolverSpScsrcholBufferInfoHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
    ccall((:cusolverSpScsrcholBufferInfoHost, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cfloat}, Ptr{Cint}, Ptr{Cint}, csrcholInfoHost_t, Ptr{Cint}, Ptr{Cint}), handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
end

@checked function cusolverSpDcsrcholBufferInfoHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
    ccall((:cusolverSpDcsrcholBufferInfoHost, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, csrcholInfoHost_t, Ptr{Cint}, Ptr{Cint}), handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
end

@checked function cusolverSpCcsrcholBufferInfoHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
    ccall((:cusolverSpCcsrcholBufferInfoHost, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{cuComplex}, Ptr{Cint}, Ptr{Cint}, csrcholInfoHost_t, Ptr{Cint}, Ptr{Cint}), handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
end

@checked function cusolverSpZcsrcholBufferInfoHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
    ccall((:cusolverSpZcsrcholBufferInfoHost, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, csrcholInfoHost_t, Ptr{Cint}, Ptr{Cint}), handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
end

@checked function cusolverSpScsrcholFactorHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pBuffer)
    ccall((:cusolverSpScsrcholFactorHost, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cfloat}, Ptr{Cint}, Ptr{Cint}, csrcholInfoHost_t, Ptr{Cvoid}), handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pBuffer)
end

@checked function cusolverSpDcsrcholFactorHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pBuffer)
    ccall((:cusolverSpDcsrcholFactorHost, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, csrcholInfoHost_t, Ptr{Cvoid}), handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pBuffer)
end

@checked function cusolverSpCcsrcholFactorHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pBuffer)
    ccall((:cusolverSpCcsrcholFactorHost, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{cuComplex}, Ptr{Cint}, Ptr{Cint}, csrcholInfoHost_t, Ptr{Cvoid}), handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pBuffer)
end

@checked function cusolverSpZcsrcholFactorHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pBuffer)
    ccall((:cusolverSpZcsrcholFactorHost, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, csrcholInfoHost_t, Ptr{Cvoid}), handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pBuffer)
end

@checked function cusolverSpScsrcholZeroPivotHost(handle, info, tol, position)
    ccall((:cusolverSpScsrcholZeroPivotHost, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, csrcholInfoHost_t, Cfloat, Ptr{Cint}), handle, info, tol, position)
end

@checked function cusolverSpDcsrcholZeroPivotHost(handle, info, tol, position)
    ccall((:cusolverSpDcsrcholZeroPivotHost, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, csrcholInfoHost_t, Cdouble, Ptr{Cint}), handle, info, tol, position)
end

@checked function cusolverSpCcsrcholZeroPivotHost(handle, info, tol, position)
    ccall((:cusolverSpCcsrcholZeroPivotHost, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, csrcholInfoHost_t, Cfloat, Ptr{Cint}), handle, info, tol, position)
end

@checked function cusolverSpZcsrcholZeroPivotHost(handle, info, tol, position)
    ccall((:cusolverSpZcsrcholZeroPivotHost, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, csrcholInfoHost_t, Cdouble, Ptr{Cint}), handle, info, tol, position)
end

@checked function cusolverSpScsrcholSolveHost(handle, n, b, x, info, pBuffer)
    ccall((:cusolverSpScsrcholSolveHost, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Ptr{Cfloat}, Ptr{Cfloat}, csrcholInfoHost_t, Ptr{Cvoid}), handle, n, b, x, info, pBuffer)
end

@checked function cusolverSpDcsrcholSolveHost(handle, n, b, x, info, pBuffer)
    ccall((:cusolverSpDcsrcholSolveHost, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Ptr{Cdouble}, Ptr{Cdouble}, csrcholInfoHost_t, Ptr{Cvoid}), handle, n, b, x, info, pBuffer)
end

@checked function cusolverSpCcsrcholSolveHost(handle, n, b, x, info, pBuffer)
    ccall((:cusolverSpCcsrcholSolveHost, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Ptr{cuComplex}, Ptr{cuComplex}, csrcholInfoHost_t, Ptr{Cvoid}), handle, n, b, x, info, pBuffer)
end

@checked function cusolverSpZcsrcholSolveHost(handle, n, b, x, info, pBuffer)
    ccall((:cusolverSpZcsrcholSolveHost, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, csrcholInfoHost_t, Ptr{Cvoid}), handle, n, b, x, info, pBuffer)
end

@checked function cusolverSpCreateCsrcholInfo(info)
    ccall((:cusolverSpCreateCsrcholInfo, CUSOLVER.libcusolver()), cusolverStatus_t, (Ptr{csrcholInfo_t},), info)
end

@checked function cusolverSpDestroyCsrcholInfo(info)
    ccall((:cusolverSpDestroyCsrcholInfo, CUSOLVER.libcusolver()), cusolverStatus_t, (csrcholInfo_t,), info)
end

@checked function cusolverSpXcsrcholAnalysis(handle, n, nnzA, descrA, csrRowPtrA, csrColIndA, info)
    ccall((:cusolverSpXcsrcholAnalysis, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cint}, Ptr{Cint}, csrcholInfo_t), handle, n, nnzA, descrA, csrRowPtrA, csrColIndA, info)
end

@checked function cusolverSpScsrcholBufferInfo(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
    ccall((:cusolverSpScsrcholBufferInfo, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cfloat}, Ptr{Cint}, Ptr{Cint}, csrcholInfo_t, Ptr{Cint}, Ptr{Cint}), handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
end

@checked function cusolverSpDcsrcholBufferInfo(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
    ccall((:cusolverSpDcsrcholBufferInfo, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, csrcholInfo_t, Ptr{Cint}, Ptr{Cint}), handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
end

@checked function cusolverSpCcsrcholBufferInfo(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
    ccall((:cusolverSpCcsrcholBufferInfo, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{cuComplex}, Ptr{Cint}, Ptr{Cint}, csrcholInfo_t, Ptr{Cint}, Ptr{Cint}), handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
end

@checked function cusolverSpZcsrcholBufferInfo(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
    ccall((:cusolverSpZcsrcholBufferInfo, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, csrcholInfo_t, Ptr{Cint}, Ptr{Cint}), handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes)
end

@checked function cusolverSpScsrcholFactor(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pBuffer)
    ccall((:cusolverSpScsrcholFactor, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cfloat}, Ptr{Cint}, Ptr{Cint}, csrcholInfo_t, Ptr{Cvoid}), handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pBuffer)
end

@checked function cusolverSpDcsrcholFactor(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pBuffer)
    ccall((:cusolverSpDcsrcholFactor, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, csrcholInfo_t, Ptr{Cvoid}), handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pBuffer)
end

@checked function cusolverSpCcsrcholFactor(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pBuffer)
    ccall((:cusolverSpCcsrcholFactor, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{cuComplex}, Ptr{Cint}, Ptr{Cint}, csrcholInfo_t, Ptr{Cvoid}), handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pBuffer)
end

@checked function cusolverSpZcsrcholFactor(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pBuffer)
    ccall((:cusolverSpZcsrcholFactor, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t, Ptr{cuDoubleComplex}, Ptr{Cint}, Ptr{Cint}, csrcholInfo_t, Ptr{Cvoid}), handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pBuffer)
end

@checked function cusolverSpScsrcholZeroPivot(handle, info, tol, position)
    ccall((:cusolverSpScsrcholZeroPivot, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, csrcholInfo_t, Cfloat, Ptr{Cint}), handle, info, tol, position)
end

@checked function cusolverSpDcsrcholZeroPivot(handle, info, tol, position)
    ccall((:cusolverSpDcsrcholZeroPivot, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, csrcholInfo_t, Cdouble, Ptr{Cint}), handle, info, tol, position)
end

@checked function cusolverSpCcsrcholZeroPivot(handle, info, tol, position)
    ccall((:cusolverSpCcsrcholZeroPivot, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, csrcholInfo_t, Cfloat, Ptr{Cint}), handle, info, tol, position)
end

@checked function cusolverSpZcsrcholZeroPivot(handle, info, tol, position)
    ccall((:cusolverSpZcsrcholZeroPivot, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, csrcholInfo_t, Cdouble, Ptr{Cint}), handle, info, tol, position)
end

@checked function cusolverSpScsrcholSolve(handle, n, b, x, info, pBuffer)
    ccall((:cusolverSpScsrcholSolve, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Ptr{Cfloat}, Ptr{Cfloat}, csrcholInfo_t, Ptr{Cvoid}), handle, n, b, x, info, pBuffer)
end

@checked function cusolverSpDcsrcholSolve(handle, n, b, x, info, pBuffer)
    ccall((:cusolverSpDcsrcholSolve, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Ptr{Cdouble}, Ptr{Cdouble}, csrcholInfo_t, Ptr{Cvoid}), handle, n, b, x, info, pBuffer)
end

@checked function cusolverSpCcsrcholSolve(handle, n, b, x, info, pBuffer)
    ccall((:cusolverSpCcsrcholSolve, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Ptr{cuComplex}, Ptr{cuComplex}, csrcholInfo_t, Ptr{Cvoid}), handle, n, b, x, info, pBuffer)
end

@checked function cusolverSpZcsrcholSolve(handle, n, b, x, info, pBuffer)
    ccall((:cusolverSpZcsrcholSolve, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, csrcholInfo_t, Ptr{Cvoid}), handle, n, b, x, info, pBuffer)
end

@checked function cusolverSpScsrcholDiag(handle, info, diag)
    ccall((:cusolverSpScsrcholDiag, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, csrcholInfo_t, Ptr{Cfloat}), handle, info, diag)
end

@checked function cusolverSpDcsrcholDiag(handle, info, diag)
    ccall((:cusolverSpDcsrcholDiag, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, csrcholInfo_t, Ptr{Cdouble}), handle, info, diag)
end

@checked function cusolverSpCcsrcholDiag(handle, info, diag)
    ccall((:cusolverSpCcsrcholDiag, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, csrcholInfo_t, Ptr{Cfloat}), handle, info, diag)
end

@checked function cusolverSpZcsrcholDiag(handle, info, diag)
    ccall((:cusolverSpZcsrcholDiag, CUSOLVER.libcusolver()), cusolverStatus_t, (cusolverSpHandle_t, csrcholInfo_t, Ptr{Cdouble}), handle, info, diag)
end
