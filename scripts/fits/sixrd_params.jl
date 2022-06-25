using CovidSim

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

start_date = Date("2020-02-11")
shift_deaths = 24

β = 0.37
μ = 1/10
c = 1.0
α = 1/18
κ = 1/2.5

tmax = 50

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

u0 = zeros(Float64, 5)
u0[1] = sum(n)

u0[1] -= sum(seed_num)
u0[2] += sum(seed_num)

########################################
# fit params
########################################

fig = Figure()
ax = Axis(fig[1, 1])

fitted_params = Float64[]

t_prev = 1
for t in phase_duration
    function op_func(β)
        p = (β, μ, α, κ, 1.0)
        prob = ODEProblem(sixrd!, u0, (0, t), p)
        sol = solve(prob)
        mse = 0
        for i in 1:t
            mse += (sol(i)[5] - df.total_deaths_shifted[t_prev + i])^2
        end

        return mse
    end

    opt = optimize(op_func, 0.0, 1.0; show_trace=true)
    fitβ = Optim.minimizer(opt)
    push!(fitted_params, fitβ)

    println(Optim.minimizer(opt))

    p = (fitβ, μ, α, κ, 1.0)
    prob = ODEProblem(sixrd!, u0, (0, t), p)
    sol = solve(prob)
    u0 .= sol[end]
    u0[5] = df.total_deaths_shifted[t_prev + t]

    lines!(ax, t_prev:(t_prev + t), [sol(x)[5] for x in 0:t])

    global t_prev += t
end

lines!(ax, df.total_deaths_shifted)
vlines!(ax, cumsum(phase_duration) .+ 1)
display(fig)
