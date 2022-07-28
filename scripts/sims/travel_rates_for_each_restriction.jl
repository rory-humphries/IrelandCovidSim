using CovidSim
using IrelandCovidSim

using StatsBase
using LinearAlgebra
using DifferentialEquations
using Distances
using JLD2
using DataFrames
using UnPack
using Dates
using GeoDataFrames
using Setfield

# for progress bar
using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

######################################
# Set parameters and load data
######################################

avg_travellers = 0.43
rm_α = 1.463
rm_β = 0.757

ed_soa_df = GeoDataFrames.read(project_path("data", "processed", "ed_soa_shapefile"))

# population vec
n = Float64.(copy(ed_soa_df.population))
# distance mat
D = zeros(nrow(ed_soa_df), nrow(ed_soa_df))
for (i, (xi, yi)) in enumerate(zip(ed_soa_df.centroid_x, ed_soa_df.centroid_y))
    for (j, (xj, yj)) in enumerate(zip(ed_soa_df.centroid_x, ed_soa_df.centroid_y))
        #D[i, j] = AG.distance(AG.centroid(ed_soa_df.geom[i]), AG.centroid(ed_soa_df.geom[j]))
        D[i, j] = euclidean([xi, yi], [xj, yj])
    end
end
D ./= 1000
# num vertices
Nv = size(n, 1)

W_vec = Vector{Matrix{Float64}}()
for max_dist in [1000, 20, 5, 2]
    # travel rate matrix
    W = zeros(Nv, Nv)
    radiation_model_matrix!(W, D, n, rm_α, rm_β)
    W[D .> max_dist] .*= 0.3

    for i in 1:size(W, 1)
        normalize!(view(W, i, :), 1)
    end
    W *= avg_travellers
    W[W .!= 0] .= -log.(1 .- W[W .!= 0])
    push!(W_vec, W)
end

hist(W_vec[1][1014, :])
hist!(W_vec[2][1014, :])
hist!(W_vec[3][1014, :])
hist!(W_vec[4][1014, :])

save(
    project_path("data", "processed", "travel_rate_mats_with_restrictions.jld2"),
    "vec",
    W_vec,
)

