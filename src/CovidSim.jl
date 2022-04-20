module CovidSim

import ArchGDAL as AG

using JLD2
using Distributions
using UnPack
using Random
using StatsBase

data_path() = normpath(joinpath(@__DIR__, "..", "data"))
export data_path

include("archgdal_serialization.jl")
export IGeometryPolygonSerialization, IGeometryMultiPolygonSerialization

include("gravity_model.jl")
export GravityModel, exp_gravity

include("lazy_matrix_mult.jl")
export LazyMatrixMult

include("random_array.jl")
export gen_sparse_array

include("sixrd.jl")
export sixrd!, accumulate_groups

include("sixrd_multiphase_lockdown.jl")
export SixrdMultiphase

end
