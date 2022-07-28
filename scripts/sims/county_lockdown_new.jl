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
end_date = Date("2021-12-31")

# parameters associated with each phase

Imax = 1400

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
#phase_duration = [3, 3, 3, 3, 3, 3, 100]
dates = start_date:Day(1):end_date

Tmax = length(dates)
# need tstops so the callback is called at the right times
tstops = 0:Tmax

ed_soa_df = GeoDataFrames.read(project_path("data", "processed", "ed_soa_shapefile"))

# county vec
county = ed_soa_df.county
# population vec
n = Float64.(copy(ed_soa_df.population))
# distance mat
D = zeros(nrow(ed_soa_df), nrow(ed_soa_df))
for (i, (xi, yi)) in enumerate(zip(ed_soa_df.centroid_x, ed_soa_df.centroid_y))
    for (j, (xj, yj)) in enumerate(zip(ed_soa_df.centroid_x, ed_soa_df.centroid_y))
        D[i, j] = euclidean([xi, yi], [xj, yj])
    end
end
D ./= 1000

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
    county=county,
    time_in_phase=zeros(Int, Nv),
    phase=ones(Int, Nv), # current phase
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
    println(sum(u; dims=1))
    println(t)
    p_ = (p.phase_β[p.phase], p.μ, p.α, p.κ, (π / 2) .* sin(2 * π * t) .* p.W)
    sixrd_metapop!(du, u, p_, t)
    du[du + u .< 0] .= 0
    return nothing
end

########################################
# The callback which updates the parameters at each step
#######################################

function update_rate_matrix!(integrator)
    @unpack p, u, t = integrator
    # update the travel rate matrix
    radiation_model_matrix!(p.W, p.D, p.n, p.rm_α, p.rm_β)

    max_dist = p.phase_max_distance[p.phase]
    p.W[(p.D .> max_dist) .| (p.D .> max_dist')] .*= (1 - p.phase_compliance)

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
    Nv = size(u, 1)

    p.time_in_phase .+= 1
    I_counties = accumulate_groups(u[:, 2] + u[:, 3], p.county)

    phase_change = false
    for v in 1:Nv
        if I_counties[p.county[v]] > p.Imax && p.phase[v] >= 6 && t > 151
            p.phase[v] = 3
            p.time_in_phase[v] = 0
            phase_change = true
        elseif p.time_in_phase[v] .>= p.phase_duration[p.phase[v]]
            if p.phase[v] < Np
                p.phase[v] += 1
            end
            p.time_in_phase[v] = 0
            phase_change = true
        end
    end
    if phase_change
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
sol = solve(prob; callback=update_params_cb, tstops=tstops, abstol=1e-2, reltol=1e-2)

sim_df = DataFrame(:date => dates)

for cnty in unique(ed_soa_df.county)
    inds = findall(x -> x == cnty, ed_soa_df.county)
    sim_df[:, cnty] = [sum(sol(t)[inds, 2:3]) for t in 1:length(dates)]
end

fig = Figure()
ax = Axis(fig[1, 1])
for cnty in unique(ed_soa_df.county)
    if cnty ∈ ["dublin", "cork", "antrim", "limerick"]
        continue
    else
        lines!(sim_df[:, cnty]; color=(:red, 0.3))
    end
end

for cnty in ["dublin", "cork", "antrim", "limerick"]
    lines!(sim_df[:, cnty]; label=cnty, linewidth=3)
end

axislegend(ax)

save(project_path("data", "processed", "county_lockdown_imax=$Imax.jld2"), "df", sim_df)

