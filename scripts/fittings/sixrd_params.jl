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
using Optim

import ArchGDAL as AG

include(joinpath(@__DIR__, "..", "..", "src", "paths.jl"))

######################################
# Parameters
######################################

# each duration is the number of days the phase lasts
phase_duration = [50, 15, 52, 21, 21, 21, 21, 21]
phase_β = [0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37]
phase_μ = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
phase_c = [1.0, 0.7, 0.4, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75]
phase_α = [0.0028, 0.0028, 0.0028, 0.0028, 0.0028, 0.0028, 0.0028, 0.0028, 0.0028]
phase_κ = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
phase_max_distance = [1000.0, 2.0, 2.0, 2.0, 5.0, 5.0, 20.0, 1000.0, 1000.0]
phase_compliance = [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]

seed_nodes = [250, 500, 1000]
seed_num = [4, 4, 4]

rm_α = 1.463
rm_β = 0.757

######################################

df = CSV.read(joinpath(data_path(), "raw", "daily-covid-cases-deaths.csv"), DataFrame)
df = df[df.Entity .== "Ireland", :]
df = df[df.Day .< Date("2021-01-01"), :]
sort!(df, :Day)

moving_average(vs, n) = [sum(@view vs[i:(i + n - 1)]) / n for i in 1:(length(vs) - (n - 1))]

df[1, "Daily new confirmed deaths due to COVID-19"] = 0
for i in findall(ismissing, df[:, "Daily new confirmed deaths due to COVID-19"])
    df[i, "Daily new confirmed deaths due to COVID-19"] = df[
        i - 1, "Daily new confirmed deaths due to COVID-19"
    ]
end

deaths_idx = Dates.value.(df.Day .- Date("2020-01-22"))[1:(end - 6)] .- 23
deaths = moving_average(df[:, "Daily new confirmed deaths due to COVID-19"], 7)

real_deaths = zeros(sum(phase_duration))
real_deaths[deaths_idx[deaths_idx .< length(real_deaths)]] .= deaths[deaths_idx .< length(
    real_deaths
)]

ed_soa_df = load(joinpath(data_path(), "processed", "ed_soa_df.jld2"), "df")

population = Float64.(copy(ed_soa_df.population))
county = ed_soa_df.county

distance = zeros(nrow(ed_soa_df), nrow(ed_soa_df))
for (i, (xi, yi)) in enumerate(zip(ed_soa_df.centroid_x, ed_soa_df.centroid_y))
    for (j, (xj, yj)) in enumerate(zip(ed_soa_df.centroid_x, ed_soa_df.centroid_y))
        distance[i, j] = haversine([xi, yi], [xj, yj])
    end
end

num_verts = size(population, 1)
num_phases = size(phase_duration, 1)

adj = zeros(num_verts, num_verts)

u0 = zeros(Float64, num_verts, 5)
u0[:, 1] = population

for i in 1:size(seed_nodes, 1)
    u0[seed_nodes[i], 2] += seed_num[i]
    u0[seed_nodes[i], 1] -= seed_num[i]
end

sim_params = (
    # vertex properties
    population=population,
    phase=1,
    time_in_phase=0,
    county=county,

    # edge properties
    distance=distance,

    # graph properties
    adj=adj,
    num_verts=num_verts,

    # phase properties
    num_phases=num_phases,
    phase_β=phase_β,
    phase_μ=phase_μ,
    phase_α=phase_α,
    phase_κ=phase_κ,
    phase_c=phase_c,
    phase_max_distance=phase_max_distance,
    phase_compliance=phase_compliance,
    phase_duration=phase_duration,

    # radiation model params,
    rm_α=rm_α,
    rm_β=rm_β,
)

########################################
# Define the ode for the multi phase county lockdown
#######################################

function sixrd_county_lockdown!(du, u, p, t=nothing)
    return sixrd!(
        du,
        u,
        (
            p.phase_β[p.phase],
            p.phase_μ[p.phase],
            p.phase_α[p.phase],
            p.phase_κ[p.phase],
            p.phase_c[p.phase],
            p.adj,
        ),
        t,
    )
end

########################################
# The callback which updates the parameters at each step
#######################################

function affect_params!(integrator)
    @unpack p, u = integrator

    p.time_in_phase .+= 1
    # node reached end of duration in phase
    if p.time_in_phase >= p.phase_duration[p.phase]
        if p.phase < p.num_phases
            p.phase += 1
            p.time_in_phase = 0
        else
            p.time_in_phase = 0
        end
    end
    return nothing
end
update_params_cb = DiscreteCallback(
    (args...) -> true, affect_params!; save_positions=(false, false)
)

########################################
# The callback which updates commuter adj matrix
#######################################

function affect_adj!(integrator)
    @unpack p, u = integrator

    c = 0.6

    adj = p.adj
    adj .= 0

    α = p.rm_α
    β = p.rm_β

    m = p.population
    n = m
    d = p.distance

    M = sum(m .^ α)
    for i in 1:(p.num_verts)
        # distancs from vert i to all others
        dvec = d[i, :]

        s = radiation_si(dvec, n .^ β)
        pvec = (1 / (1 - (m[i]^α / M))) .* radiation_model(m[i]^α, n .^ β, s)

        # probability changed if outside max distance
        max_dist = p.phase_max_distance[p.phase]
        pvec[dvec .> max_dist] .*= 1 .- p.phase_compliance[p.phase]

        num_out = round(Int, c * m[i], RoundDown)
        j = sample(1:(p.num_verts), Weights(pvec), num_out)

        for k in j
            adj[i, k] += 1
        end
    end
    return nothing
end
update_adj_cb = DiscreteCallback(
    (args...) -> true, affect_adj!; save_positions=(false, false)
)

########################################
# Run the sim
########################################

prob = DiscreteProblem(sixrd_county_lockdown!, u0, (0, sum(phase_duration)), sim_params)
integrator = init(prob, FunctionMap{true}())

sol = solve(
    prob,
    FunctionMap{true}();
    callback=CallbackSet(update_adj_cb, update_params_cb),
    progress=true,
    progress_steps=1,
)

fig = Figure()
ax = Axis(fig[1, 1])#, yscale=log10)
lines!(ax, diff(vec(sum(sol[:, 5, :]; dims=1))))
vlines!(ax, cumsum(phase_duration) .+ 1)

lines!(ax)
