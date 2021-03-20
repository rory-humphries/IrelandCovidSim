module CovidSim

include("RandomArray.jl")

export gen_sparse_array

include("SIXRD.jl")

export SIXRDMetaPopParams, SIXRDMetaPopODE, accumulate_groups


end