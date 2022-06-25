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
using Dates

# for progress bar
using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

include(joinpath(@__DIR__, "..", "..", "src", "paths.jl"))

# Parameters
######################################

start_duration = 21

school_close_date = Date("2020-03-12")
start_date = Date("2020-03-12") - Day(start_duration)

stay_at_home = Date("2020-03-27")
phase_1 = Date("2020-05-18")
phase_2 = Date("2020-06-08")
phase_3 = Date("2020-06-29")
phase_4_paused = Date("2020-07-20")
phase_4 = Date("2020-09-21")

# each duration is the number of days the phase lasts
phase_duration = [
    Dates.value(school_close_date - start_date),
    Dates.value(stay_at_home - school_close_date),
    Dates.value(phase_1 - stay_at_home),
    Dates.value(phase_2 - phase_1),
    Dates.value(phase_3 - phase_2),
    Dates.value(phase_4_paused - phase_3),
    Dates.value(phase_4 - phase_4_paused),
]
phase_β = [0.49, 0.37, 0.1315, 0.195, 0.195, 0.195, 0.195, 0.197, 0.197]
phase_μ = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
phase_c = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
phase_α = [0.0028, 0.0028, 0.0028, 0.0028, 0.0028, 0.0028, 0.0028, 0.0028, 0.0028]
phase_κ = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
phase_max_distance = [1000.0, 2.0, 2.0, 2.0, 5.0, 5.0, 20.0, 1000.0, 1000.0]
phase_compliance = [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]

tstops = append!([0.001], cumsum(phase_duration) .+ 1)

Tmax = tstops[7] + 1
Imax = 1400

seed_nodes = [250, 500, 1000]
seed_num = [4, 4, 4]

rm_α = 1.463
rm_β = 0.757

########################################
# Define the ode for the multi phase county lockdown
#######################################

function sim_ode!(du, u, p, t)
    pnew = (
        p.phase_β[p.phase],
        p.phase_μ[p.phase],
        p.phase_α[p.phase],
        p.phase_κ[p.phase],
        p.phase_c[p.phase],
        (π / 2) .* sin(2 * π * t) .* p.adj,
    )
    return sixrd_metapop!(du, u, pnew, t)
end

########################################
# The callback which updates the parameters at each step
#######################################

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
    phase=ones(Int, num_verts),
    time_in_phase=zeros(Int, num_verts),
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

    # cutoff number of infected for entering lockdown
    Imax=Imax,

    # radiation model params,
    rm_α=rm_α,
    rm_β=rm_β,
)

function affect_params!(integrator)
    @unpack p, u = integrator

    p.phase .+= 1
    """
    p.time_in_phase .+= 1
    I_counties = accumulate_groups(u[:, 2], p.county)

    for v in 1:(p.num_verts)

        # node belongs to county over Imax
        if I_counties[p.county[v]] > p.Imax && p.phase[v] > 4 && integrator.t > 222
            p.phase[v] = 4
            p.time_in_phase[v] = 0
            # node reached end of duration in phase
        elseif p.time_in_phase[v] >= p.phase_duration[p.phase[v]]
            if p.phase[v] < p.num_phases
                p.phase[v] += 1
                p.time_in_phase[v] = 0
            else
                p.time_in_phase[v] = 0
            end
        end
    end
    """
    return nothing
end
update_params_cb = DiscreteCallback(
    (u, t, integrator) -> t ∈ tstops[2:end], affect_params!; save_positions=(false, false)
)

########################################
# The callback which updates commuter adj matrix
#######################################

function affect_adj!(integrator)
    @unpack p, u = integrator
    println("affect adj here and t=", integrator.t)

    c = 0.6

    W = p.adj
    W .= 0

    α = p.rm_α
    β = p.rm_β

    max_dist = p.phase_max_distance[p.phase]

    n = p.population
    Nv = length(n)
    d = p.distance

    for i in 1:Nv
        # distancs from vert i to all others
        di = d[i, :]

        s = radiation_si(di, n .^ β)
        pvec = radiation_model(n[i]^α, n .^ β, s)
        pvec[di .> min.(max_dist, max_dist[i])] *= 0.3
        pvec[i] = 0
        normalize!(pvec, 1)
        W[i, :] .= pvec
    end

    W *= c
    W[W .!= 0] .= -log.(1 .- W[W .!= 0])
    return nothing
end
update_adj_cb = DiscreteCallback(
    (u, t, integrator) -> t ∈ tstops, affect_adj!; save_positions=(false, false)
)

########################################
# Run the sim
########################################

prob = ODEProblem(sim_ode!, u0, (0.0, Float64(Tmax)), sim_params)
sol = solve(
    prob;
    callback=CallbackSet(update_adj_cb, update_params_cb),
    progress=true,
    progress_steps=1,
    tstops=tstops,
)

fig = Figure()
ax = Axis(fig[1, 1])#, yscale=log10)

lentime = length(1:Tmax)
slice_dates = range(1, lentime; step=lentime ÷ 8)

ax.xticks = (slice_dates, string.(dates[slice_dates]))
ax.xticklabelrotation = π / 4
ax.xticklabelalign = (:right, :center)

lines!(ax, 1:Tmax, [sum(sol(t)[:, 5]) for t in 1:Tmax])
vlines!(ax, cumsum(phase_duration) .+ 1)
lines!(ax, total_deaths_shifted)
lines!(ax, total_deaths)

