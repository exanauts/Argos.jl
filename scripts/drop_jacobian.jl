

using Argos
using MadNLP

datafile = "../matpower/data/case9.m"

#=
Solve OPF in full-space
=#
flp = Argos.FullSpaceEvaluator(datafile)
opf_full = Argos.OPFModel(flp)
madnlp(opf_full)

#=
Solve OPF in reduced-space and get evaluation of Jacobian
=#
rlp = Argos.ReducedSpaceEvaluator(datafile)
opf_reduced = Argos.OPFModel(rlp)
madnlp(opf_reduced)

# Get evolution of Jacobian
Gxs = opf_reduced.etc[:powerflow_jacobian]
Ginit = Gxs[1]
Gfinal = Gxs[end]
