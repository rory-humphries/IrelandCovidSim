# model multiple covid-19 outbreak from a single infected staring in different locations
#

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
end_date = start_date + Day(200)

# parameters associated with each phase

β = 0.687
μ = (1 - 0.03) * (1 / 10)
α = (0.03) * (1 / 10)
κ = 1 / 2.5

rm_α = 1.463
rm_β = 0.757

avg_travellers = 0.43

dates = start_date:Day(1):end_date

Tmax = length(dates)

ed_soa_df = GeoDataFrames.read(project_path("data", "processed", "ed_soa_shapefile"))

# population vec
n = Float64.(copy(ed_soa_df.population))
# distance mat
D = zeros(nrow(ed_soa_df), nrow(ed_soa_df))
for (i, (xi, yi)) in enumerate(zip(ed_soa_df.centroid_x, ed_soa_df.centroid_y))
    for (j, (xj, yj)) in enumerate(zip(ed_soa_df.centroid_x, ed_soa_df.centroid_y))
        D[i, j] = haversine([xi, yi], [xj, yj])
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

# named tuple to hold all data for diff eq solver
sim_params = (
    n=n, # population vector
    D=D, # distance matrix
    W=W, # travel rate matrix
    μ=μ, # rate of recovery
    α=α, # rate of death
    κ=κ, # rate of quarantine
    β=β, # rate of infection
)
########################################
# Define the ode for the multi phase lockdown
#######################################

function sim_ode!(du, u, p, t)
    p_ = (p.β, p.μ, p.α, p.κ, (π / 2) .* sin(2 * π * t) .* p.W)
    return sixrd_metapop!(du, u, p_, t)
end

########################################
# Run the sim
########################################

df_vec = []

fig = Figure()
ax = Axis(fig[1,1])

for i in 1:10
    # initial conditions
    u0 = zeros(Float64, Nv, 5)
    v = sample(1:Nv)
    u0[:, 1] .= n
    u0[v, 1] -= 1
    u0[v, 2] += 1

    prob = ODEProblem(sim_ode!, u0, (0.0, Float64(Tmax)), sim_params)
    sol = solve(prob; progress=true, progress_steps=1)

    sim_df = DataFrame(:date => dates)
    sim_df.S = [sum(sol(t)[:, 1]) for t in 0:length(dates)-1]
    sim_df.I = [sum(sol(t)[:, 2]) for t in 0:length(dates)-1]
    sim_df.Q = [sum(sol(t)[:, 3]) for t in 0:length(dates)-1]
    sim_df.R = [sum(sol(t)[:, 4]) for t in 0:length(dates)-1]
    sim_df.D = [sum(sol(t)[:, 5]) for t in 0:length(dates)-1]

    push!(df_vec, sim_df)
    lines!(ax, sim_df.I + sim_df.Q)

end
    
lines(df_vec[1].I + df_vec[1].Q, color=:red)
for df in df_vec[2:end]
    lines!(df.I + df.Q, color=:red)
end

save(project_path("data", "processed", "no_lockdown_multiple_runs.jld2"), "vec", df_vec)

#save(project_path("data", "processed", "initial_outbreak_reopen_full.jld2"), "sol", sol)

