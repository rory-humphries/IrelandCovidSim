using CovidSim

using DelimitedFiles
using StatsBase
using Random
using Distributions
using SparseArrays
using LinearAlgebra
using DifferentialEquations
using Distances
using JLD2
using ArchGDAL
using DataFrames
using UnPack

# for progress bar
using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

include(joinpath(@__DIR__, "..", "..", "src", "paths.jl"))

######################################
# Parameters
######################################

β = 0.37
μ = 0.1
c = 1.0
α = 0.0028
κ = 0.1

Tmax = 50

seed_nodes = [250, 500, 1000]
seed_num = [4, 4, 4]

rm_α = 1.463
rm_β = 0.757

######################################

ed_soa_df = load(joinpath(data_path(), "processed", "ed_soa_df.jld2"), "df")

n = Float64.(copy(ed_soa_df.population))
Nv = size(n, 1)

d = zeros(nrow(ed_soa_df), nrow(ed_soa_df))
for (i, (xi, yi)) in enumerate(zip(ed_soa_df.centroid_x, ed_soa_df.centroid_y))
    for (j, (xj, yj)) in enumerate(zip(ed_soa_df.centroid_x, ed_soa_df.centroid_y))
        d[i, j] = haversine([xi, yi], [xj, yj])
    end
end

W = zeros(Nv, Nv)
for i in 1:Nv
    # distancs from vert i to all others
    di = d[i, :]

    s = radiation_si(di, n .^ β)
    p = radiation_model(n[i]^α, n .^ β, s)
    p[i] = 0
    normalize!(p, 1)
    W[i, :] .= p
end

W *= 0.6
W[W .!= 0] .= -log.(1 .- W[W .!= 0])

p = (β, μ, α, κ, c, W)

u0 = zeros(Float64, Nv, 5)
u0[:, 1] = n

for i in 1:size(seed_nodes, 1)
    u0[seed_nodes[i], 2] += seed_num[i]
    u0[seed_nodes[i], 1] -= seed_num[i]
end

########################################
# Run the sim
########################################

function sim_ode!(du, u, p, t)
    pnew = (p[1:5]..., (π/2)*sin(2*π*t)*p[6])
    return sixrd_metapop!(du, u, pnew, t)
end

prob = ODEProblem(sim_ode!, u0, (0, Tmax), p)
sol = solve(prob; progress=true, progress_steps=1)

using GLMakie

fig = Figure()
ax = Axis(fig[1, 1])#, yscale=log10)
lines!(ax, vec(sum(sol[:, 1, :]; dims=1)))
lines!(ax, vec(sum(sol[:, 2, :]; dims=1)))
lines!(ax, vec(sum(sol[:, 3, :]; dims=1)))
lines!(ax, vec(sum(sol[:, 4, :]; dims=1)))
lines!(ax, vec(sum(sol[:, 5, :]; dims=1)))
