using CovidSim
using GLMakie
using DataFrames
using JLD2
using InlineStrings
using Distances
using StatsBase
using LinearAlgebra

import ArchGDAL as AG

include(joinpath(@__DIR__, "..", "..", "src", "paths.jl"))

########################################
# parameters and data
########################################

#α = 1.463
#β = 0.757
α = 1.187
β = 0.831
N = 5 * 10^4

########################################
# load data
########################################

ed_soa_df = load(joinpath(data_path(), "processed", "ed_soa_df.jld2"), "df")
ed_travels_df = load(joinpath(data_path(), "processed", "ed_travels_df.jld2"), "df")

# remove commuting with distance 0
delete!(ed_travels_df, ed_travels_df.distance .== 0)

sample_df = load(
    joinpath(
        data_path(), "processed", "radiation_model_sims", "roi_α=$(α)_β=$(β)_N=$(N).jld2"
    ),
    "df",
)

########################################
# plot
########################################

fig = Figure()

ax = Axis(
    fig[1, 1];
    xscale=log10,
    yscale=log10,
    xminorticksvisible=true,
    xminorgridvisible=true,
    xminorticks=IntervalsBetween(9),
    yminorticksvisible=true,
    yminorgridvisible=true,
    yminorticks=IntervalsBetween(9),
    xlabel="distance [km]",
    ylabel="pdf",
)

edges = exp10.(range(log10(1), log10(maximum(ed_travels_df.distance)), 20))

h1 = normalize(fit(Histogram, ed_travels_df.distance, edges); mode=:pdf)
h2 = normalize(fit(Histogram, sample_df.distance ./ 1000, edges); mode=:pdf)

scatterlines!(midpoints(h1.edges[1]), h1.weights)
scatterlines!(midpoints(h2.edges[1])[h2.weights .!= 0], h2.weights[h2.weights .!= 0])

display(fig)
