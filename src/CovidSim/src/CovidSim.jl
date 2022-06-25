module CovidSim

using JLD2
using Distributions
using UnPack
using Random
using StatsBase
using SparseArrays
using LinearAlgebra


import ArchGDAL as AG

include("archgdal_serialization.jl")
export IGeometryPolygonSerialization, IGeometryMultiPolygonSerialization

include("radiation_model.jl")
export radiation_model, radiation_sij, radiation_si

include("sixrd.jl")
export sixrd!, sixrd_metapop!, accumulate_groups

end
