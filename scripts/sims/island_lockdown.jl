# Model the initial outbreak of COVID-19 in Ireland from some days before the first
# reported case to the end of phase 3 reopening

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

# start date of each lockdown "phase"
start_date = Date("2020-02-20")
school_close_date = Date("2020-03-12")
stay_at_home = Date("2020-03-27")
phase_1 = Date("2020-05-18")
phase_2 = Date("2020-06-08")
phase_3 = Date("2020-06-29")
reopen = Date("2020-07-20")
end_date = Date("2022-12-31")

# parameters associated with each phase

Imax = 1000

phase_β = [0.687, 0.55, 0.43, 0.49, 0.49, 0.49, 0.687]
μ = (1 - 0.03) * (1 / 10)
α = (0.03) * (1 / 10)
κ = 1 / 2.5

phase_compliance = 0.7
phase_max_distance = [Inf, 2.0, 2.0, 5.0, 20.0, Inf, Inf]

rm_α = 1.463
rm_β = 0.757

avg_travellers = 0.43

phase_duration = [
    Dates.value(school_close_date - start_date),
    Dates.value(stay_at_home - school_close_date),
    Dates.value(phase_1 - stay_at_home),
    Dates.value(phase_2 - phase_1),
    Dates.value(phase_3 - phase_2),
    Dates.value(reopen - phase_3),
    Dates.value(end_date - reopen),
]
dates = start_date:Day(1):end_date

Tmax = length(dates)
# need tstops so the callback is called at the right times
tstops = 0:Tmax

ed_soa_df = GeoDataFrames.read(project_path("data", "processed", "ed_soa_shapefile"))

# population vec
n = Float64.(copy(ed_soa_df.population))
# distance mat
D = zeros(nrow(ed_soa_df), nrow(ed_soa_df))
for (i, (xi, yi)) in enumerate(zip(ed_soa_df.centroid_x, ed_soa_df.centroid_y))
    for (j, (xj, yj)) in enumerate(zip(ed_soa_df.centroid_x, ed_soa_df.centroid_y))
        D[i, j] = euclidean([xi, yi], [xj, yj])
    end
end

# num vertices
Nv = size(n, 1)
# num phases
Np = size(phase_duration, 1)

# travel rate matrix
W = zeros(Nv, Nv)
radiation_model_matrix!(W, D, n, rm_α, rm_β)
for i in 1:size(W, 1)
    normalize!(view(W, i, :), 1)
end
W *= avg_travellers
W[W .!= 0] .= -log.(1 .- W[W .!= 0])

# initial conditions
u0 = zeros(Float64, Nv, 5)
u0[:, 1] .= n .- 0.01
u0[:, 2] .= 0.01

# named tuple to hold all data for diff eq solver
sim_params = (
    n=n, # population vector
    time_in_phase=Ref{Int}(0),
    phase=Ref{Int}(1), # current phase
    Imax=Imax,
    D=D, # distance matrix
    W=W, # travel rate matrix
    μ=μ, # rate of recovery
    α=α, # rate of death
    κ=κ, # rate of quarantine
    phase_β=phase_β, # rate of infection for each phase
    phase_max_distance=phase_max_distance,# max distance at each phase
    phase_compliance=phase_compliance,# compliance with max dist at each phase
    phase_duration=phase_duration,# duration of each phase
    rm_α=rm_α, # exponent in radiation model for source pop
    rm_β=rm_β, # exponent in radiation model for dest pop
    avg_travellers=avg_travellers,# avg proportion of travellers out of each vertex
)
########################################
# Define the ode for the multi phase lockdown
#######################################

function sim_ode!(du, u, p, t)
    p_ = (p.phase_β[p.phase[]], p.μ, p.α, p.κ, (π / 2) .* sin(2 * π * t) .* p.W)
    return sixrd_metapop!(du, u, p_, t)
end

########################################
# The callback which updates the parameters at each step
#######################################

function update_rate_matrix!(integrator)
    @unpack p, u, t = integrator

    # update the travel rate matrix
    radiation_model_matrix!(p.W, p.D, p.n, p.rm_α, p.rm_β)

    max_dist = p.phase_max_distance[p.phase[]]
    p.W[p.D .> max_dist] .*= (1 - p.phase_compliance)

    # normalize each row
    for i in 1:size(p.W, 1)
        normalize!(view(p.W, i, :), 1)
    end
    p.W .*= p.avg_travellers
    p.W[p.W .!= 0] .= -log.(1 .- p.W[p.W .!= 0])

    return nothing
end

function update_phase_params!(integrator)
    @unpack p, u, t = integrator

    Np = length(p.phase_duration)

    p.time_in_phase[] += 1

    Isum = sum(u[:, 2] + u[:, 3])

    if Isum > p.Imax && p.phase[] >= 6
        p.phase[] = 3
        p.time_in_phase[] = 0
        update_rate_matrix!(integrator)
    elseif p.time_in_phase[] .>= p.phase_duration[p.phase[]]
        if p.phase[] < Np
            p.phase[] += 1
        end
        p.time_in_phase[] = 0
        update_rate_matrix!(integrator)
    end
end

update_params_cb = DiscreteCallback(
    (u, t, integrator) -> t ∈ tstops, update_phase_params!; save_positions=(false, false)
)

########################################
# Run the sim
########################################

prob = ODEProblem(sim_ode!, u0, (0.0, Float64(Tmax)), sim_params)
sol = solve(prob; callback=update_params_cb, progress=true, progress_steps=1, tstops=tstops)

sim_df = DataFrame(:date => dates)
sim_df.S = [sum(sol(t)[:, 1]) for t in 1:length(dates)]
sim_df.I = [sum(sol(t)[:, 2]) for t in 1:length(dates)]
sim_df.Q = [sum(sol(t)[:, 3]) for t in 1:length(dates)]
sim_df.R = [sum(sol(t)[:, 4]) for t in 1:length(dates)]
sim_df.D = [sum(sol(t)[:, 5]) for t in 1:length(dates)]

save(project_path("data", "processed", "island_lockdown_imax=$Imax.jld2"), "df", sim_df)

