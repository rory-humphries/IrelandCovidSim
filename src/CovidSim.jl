module CovidSim
using JLD2
import ArchGDAL as AG

data_path() = normpath(joinpath(@__DIR__, "..", "data"))
export data_path

include("archgdal_serialization.jl")
export IGeometryPolygonSerialization, IGeometryMultiPolygonSerialization

include("RandomArray.jl")

export gen_sparse_array

include("SIXRD.jl")

export SIXRDMetaPopParams, SIXRDMetaPopODE, accumulate_groups

end
