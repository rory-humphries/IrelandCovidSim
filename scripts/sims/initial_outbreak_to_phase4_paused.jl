using CovidSim

using StatsBase
using LinearAlgebra
using DifferentialEquations
using Distances
using JLD2
using DataFrames
using UnPack
using Dates

# for progress bar
using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

# Parameters
######################################

start_date = Date("2020-02-20")
school_close_date = Date("2020-03-12")
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
0.37755809966335657
0.36475333471785015
0.2848797404591351
0.2746794587758346
5.811830337926569e-15
0.572855700324278
1.0393184252234168e-15

phase_β = [
    0.687,
    0.672,
    0.49,
    0.51,
    0.51,
    0.51,
    0.51,
]
phase_μ = repeat([1 / 10], length(phase_β))
phase_c = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
phase_α = repeat([1 / 18], length(phase_β))
phase_κ = repeat([1 / 2.5], length(phase_β))
phase_max_distance = [1000.0, 2.0, 2.0, 5.0, 20.0, 1000.0, 20.0, 1000.0, 1000.0]
phase_compliance = [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]

tstops = append!([0.001], cumsum(phase_duration) .+ 1)

Tmax = sum(phase_duration)

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

ed_soa_df = GeoDataFrames.read(project_path("data", "processed", "ed_soa_shapefile"))

population = Float64.(copy(ed_soa_df.population))
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

function affect_params!(integrator)
    @unpack p, u = integrator

    p.phase .+= 1
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

    c = 0.43

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

sim_df = DataFrame(:date => dates)
sim_df.S = [sum(sol(t)[:, 1]) for t in 1:length(dates)]
sim_df.I = [sum(sol(t)[:, 2]) for t in 1:length(dates)]
sim_df.Q = [sum(sol(t)[:, 3]) for t in 1:length(dates)]
sim_df.R = [sum(sol(t)[:, 4]) for t in 1:length(dates)]
sim_df.D = [sum(sol(t)[:, 5]) for t in 1:length(dates)]

save(
    project_path("data", "processed", "initial_lockdown_to_phase4_paused.jld2"),
    "df",
    sim_df,
)
