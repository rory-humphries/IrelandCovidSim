using Thesis

using StatsBase
using SparseArrays
using LinearAlgebra
using DifferentialEquations
using Distances
using JLD2
using DataFrames
using UnPack
using CSV
using Dates
using Optim
using GLMakie
using GeoDataFrames

# for progress bar
using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())


######################################
# Parameters
######################################

shift_deaths = 24

β = 0.37
μ = 1/10
c = 1.0
α = 1/18
κ = 1/2.5

seed_nodes = [250, 500, 1000]
seed_num = [4, 4, 4]

rm_α = 1.463
rm_β = 0.757

######################################

start_date = Date("2020-02-20")
school_close_date = Date("2020-03-12")

stay_at_home = Date("2020-03-27")
phase_1 = Date("2020-05-18")
phase_2 = Date("2020-06-08")
phase_3 = Date("2020-06-29")
phase_4_paused = Date("2020-07-20")
phase_4 = Date("2020-09-21")

phase_duration = [
    Dates.value(school_close_date - start_date),
    Dates.value(stay_at_home - school_close_date),
    Dates.value(phase_1 - stay_at_home),
    Dates.value(phase_2 - phase_1),
    Dates.value(phase_3 - phase_2),
    Dates.value(phase_4_paused - phase_3),
    Dates.value(phase_4 - phase_4_paused),
]
phase_max_distance = [1000.0, 2.0, 2.0, 5.0, 20.0, 1000.0, 20.0, 1000.0, 1000.0]

end_date = start_date + Day(sum(phase_duration))
dates = start_date:Day(1):end_date

df = load(project_path("data", "processed", "covid_data_from_$(start_date).jld2"), "df")
delete!(df, (start_date .< df.date) .& (df.date .> end_date))

df.total_deaths_shifted = zeros(Int, length(dates))
df.total_deaths_shifted[1:(end - shift_deaths)] .= df.total_deaths[(shift_deaths + 1):end]
last_nz_id = findlast(!iszero, df.total_deaths_shifted)
df.total_deaths_shifted[last_nz_id:end] .= df.total_deaths_shifted[last_nz_id]

ed_soa_df = GeoDataFrames.read(project_path("data", "processed", "ed_soa_shapefile"))

n = Float64.(copy(ed_soa_df.population))
Nv = size(n, 1)

d = zeros(nrow(ed_soa_df), nrow(ed_soa_df))
for (i, (xi, yi)) in enumerate(zip(ed_soa_df.centroid_x, ed_soa_df.centroid_y))
    for (j, (xj, yj)) in enumerate(zip(ed_soa_df.centroid_x, ed_soa_df.centroid_y))
        d[i, j] = haversine([xi, yi], [xj, yj])
    end
end

u0 = zeros(Float64, Nv, 5)
u0[:, 1] = n

for i in 1:size(seed_nodes, 1)
    u0[seed_nodes[i], 2] += seed_num[i]
    u0[seed_nodes[i], 1] -= seed_num[i]
end

########################################
# fit params
########################################
function sim_ode!(du, u, p, t)
    pnew = (p[1:5]..., (π / 2) .* sin(2 * π * t) .* p[6])
    return sixrd_metapop!(du, u, pnew, t)
end

β_list = Float64[]
t_prev = 1
for ind in eachindex(phase_duration)
    t = phase_duration[ind]
    max_dist = phase_max_distance[ind]

    W = zeros(Nv, Nv)
    for i in 1:Nv
        # distancs from vert i to all others
        di = d[i, :]

        s = radiation_si(di, n .^ rm_β)
        p = radiation_model(n[i]^rm_α, n .^ rm_β, s)
        p[di .> max_dist] *= 0.3
        p[i] = 0
        normalize!(p, 1)
        W[i, :] .= p
    end

    W *= 0.43
    W[W .!= 0] .= -log.(1 .- W[W .!= 0])

    function op_func(β)
        p = (β, μ, α, κ, 1.0, W)
        prob = ODEProblem(sim_ode!, u0, (0.0, Float64(t)), p)
        sol = solve(prob; progress=true, progress_steps=1)
        mse = 0
        for i in 1:t
            mse += (sum(sol(i)[:, 5]) - df.total_deaths_shifted[t_prev + i])^2
        end

        fig = Figure()
        ax = Axis(fig[1, 1])

        lines!(ax, df.total_deaths_shifted)
        lines!(ax, t_prev:(t_prev + t), [sum(sol(x)[:, 5]) for x in 0:t])
        vlines!(ax, cumsum(phase_duration) .+ 1)
        display(fig)

        return mse
    end

    opt = optimize(op_func, 0.0, 1.0; show_trace=true, rel_tol=1e-2)
    fitβ = Optim.minimizer(opt)
    push!(β_list, fitβ)
    println(Optim.minimizer(opt))

    p = (β_list[ind], μ, α, κ, 1.0, W)
    prob = ODEProblem(sim_ode!, u0, (0, t), p)
    sol = solve(prob; progress=true, progress_steps=1)
    u0 .= sol[end]

    t_prev += t
end
