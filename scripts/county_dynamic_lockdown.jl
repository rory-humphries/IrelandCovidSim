using DelimitedFiles
using StatsBase
using Random
using Distributions
using SparseArrays
using LinearAlgebra
using DifferentialEquations
using Plots
using Distances
using JLD2
using ArchGDAL
using DataFrames

using CovidSim

# Parameters
######################################

# each duration is the number of days the phase lasts
const phase_duration = [50, 15, 52, 21, 21, 21, 21, 21, 400]
const phase_β = [0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37]
const phase_μ = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
const phase_c = [1.0, 0.7, 0.4, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75]
const phase_α = [0.0028, 0.0028, 0.0028, 0.0028, 0.0028, 0.0028, 0.0028, 0.0028, 0.0028]
const phase_κ = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
const phase_max_distance = [1000.0, 2.0, 2.0, 2.0, 5.0, 5.0, 20.0, 1000.0, 1000.0]
const phase_compliance = [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]
const tmax = 200
const Imax = 1400

const seed_nodes = [250, 500, 1000]
const seed_num = [4, 4, 4]

const gm_p = (0.5212685415091278, 0.7486233175103572, 8.549615110124643e-5)

function grav_func(ui, uj, v, p)
    (α, β, γ, comp) = p
    attr_i = ui[1]
    attr_j = uj[1]
    max_dist_i = ui[2]
    max_dist_j = uj[2]
    dist = v[1]

    if max_dist_i < dist || max_dist_j < dist
        return attr_i^α * attr_j^β * exp(-γ * dist) * (1 - comp)
    else
        return attr_i^α * attr_j^β * exp(-γ * dist)
    end
end

########################################
# Define parameter struct for the ode
########################################

struct SixrdCountyLockdownParams{Pt,Ft,GPt}
    # sixrd params
    sixrd_p::Pt

    # phase params
    phase_β::Vector{Float64}
    phase_μ::Vector{Float64}
    phase_α::Vector{Float64}
    phase_κ::Vector{Float64}
    phase_c::Vector{Float64}
    phase_max_distance::Vector{Float64}
    phase_compliance::Vector{Float64}
    phase_duration::Vector{Int64}

    # node and network properties
    population::Vector{Float64}
    counties::Vector{String}
    cur_duration::Vector{Int64}
    cur_max_dist::Vector{Float64}
    cur_compliance::Vector{Float64}
    cur_phase::Vector{Int64}

    # edge properties
    distances::Matrix{Float64}

    # others
    Imax::Int64
    grav_func::Ft
    grav_p::GPt
end

function SixrdCountyLockdownParams(
    sixrd_p::Pt,
    phase_β::Vector{Float64},
    phase_μ::Vector{Float64},
    phase_α::Vector{Float64},
    phase_κ::Vector{Float64},
    phase_c::Vector{Float64},
    phase_max_distance::Vector{Float64},
    phase_compliance::Vector{Float64},
    phase_duration::Vector{Int},
    population::Vector{Float64},
    counties::Vector{String},
    distances::Matrix{Float64},
    Imax::Int64,
    grav_func::Ft,
    grav_p::GPt,
) where {Pt,Ft,GPt}
    Nv = length(counties)
    cur_duration = ones(Int64, Nv) * phase_duration[1]
    cur_max_dist = ones(Float64, Nv) * phase_max_distance[1]
    cur_compliance = ones(Float64, Nv) * phase_compliance[1]
    cur_phase = ones(Int64, Nv)

    return SixrdCountyLockdownParams(
        sixrd_p,
        phase_β,
        phase_μ,
        phase_α,
        phase_κ,
        phase_c,
        phase_max_distance,
        phase_compliance,
        phase_duration,
        population,
        counties,
        cur_duration,
        cur_max_dist,
        cur_compliance,
        cur_phase,
        distances,
        Imax,
        grav_func,
        grav_p,
    )
end

########################################
# Define the ode for the multi phase county lockdown
#######################################

function sixrd_county_lockdown!(du, u, p, t=nothing)
    return sixrd!(du, u, p.sixrd_p, t)
end

########################################
# The callback which updates the parameters at each step
#######################################

function affect_params1!(integrator)
    #Nv = size(integrator.u, 1)
    Np = size(integrator.p.phase_duration, 1)
    # IJ = rand(cb.spl, 10000)

    #integrator.p.sixrd_p[6] .= sparse(IJ[1, :], IJ[2, :], ones(10000), Nv, Nv)
    integrator.p.cur_duration .-= 1

    I_counties = accumulate_groups(integrator.u[:, 2], integrator.p.counties)

    for v in 1:num_verts

        # node belongs to county over Imax
        if I_counties[integrator.p.counties[v]] > integrator.p.Imax &&
            integrator.p.cur_phase[v] > 3
            integrator.p.cur_phase[v] = 3
            integrator.p.cur_duration[v] = integrator.p.phase_duration[integrator.p.cur_phase[v]]
            # node reached end of duration in phase
        elseif integrator.p.cur_duration[v] == 0
            if integrator.p.cur_phase[v] <= Np
                integrator.p.cur_phase[v] += 1
                integrator.p.cur_duration[v] = integrator.p.phase_duration[integrator.p.cur_phase[v]]
            else
                integrator.p.cur_duration[v] = integrator.p.phase_duration[integrator.p.cur_phase[v]]
            end
        end
    end

    integrator.p.sixrd_p[1] .= integrator.p.phase_β[integrator.p.cur_phase]
    integrator.p.sixrd_p[2] .= integrator.p.phase_μ[integrator.p.cur_phase]
    integrator.p.sixrd_p[3] .= integrator.p.phase_α[integrator.p.cur_phase]
    integrator.p.sixrd_p[4] .= integrator.p.phase_κ[integrator.p.cur_phase]
    integrator.p.sixrd_p[5] .= integrator.p.phase_c[integrator.p.cur_phase]
    return nothing
end
param_update_cb = DiscreteCallback(
    (args...) -> true, affect_params1!; save_positions=(false, false)
)

########################################
# The callback which updates commuter adj matrix
#######################################

function affect_adj!(integrator)
    N = size(integrator.u, 1)
    u = cat(integrator.p.population', integrator.p.cur_max_dist'; dims=1)
    println(size(u))
    v = reshape(integrator.p.distances, (1, N, N))
    gm = GravityModel(u, v, integrator.p.grav_func, (integrator.p.grav_p..., 0.7))

    spl = sampler(gm)
    IJ = rand(spl, 10000)
    integrator.p.sixrd_p[6] .= sparse(IJ[1, :], IJ[2, :], ones(10000), N, N)
    display(integrator.p.sixrd_p[6])
    return nothing
end
update_adj_cb = DiscreteCallback(
    (args...) -> true, affect_adj!; save_positions=(false, false)
)

########################################
# Run the sim
########################################

const ed_soa_df = load(joinpath(data_path(), "processed", "ed_soa_df.jld2"), "df")

const population = Float64.(copy(ed_soa_df.population))
const distances = zeros(nrow(ed_soa_df), nrow(ed_soa_df))
const counties = ed_soa_df.county
for (i, (xi, yi)) in enumerate(zip(ed_soa_df.centroid_x, ed_soa_df.centroid_y))
    for (j, (xj, yj)) in enumerate(zip(ed_soa_df.centroid_x, ed_soa_df.centroid_y))
        distances[i, j] = haversine([xi, yi], [xj, yj])
    end
end

const N = length(population)
const num_verts = size(population, 1)
const num_phases = size(phase_duration, 1)

const u = copy(population')
const v = reshape(distances, (1, N, N))

const gm = GravityModel(u, v, exp_gravity, gm_p)
const spl = sampler(gm)

const IJ = rand(spl, 10000)
const adj = sparse(IJ[1, :], IJ[2, :], ones(10000), N, N)

const u0 = zeros(Float64, N, 5)
u0[:, 1] = population

for i in 1:size(seed_nodes, 1)
    u0[seed_nodes[i], 2] += seed_num[i]
    u0[seed_nodes[i], 1] -= seed_num[i]
end

const sixrd_p = (
    phase_β[1] * ones(N),
    phase_μ[1] * ones(N),
    phase_α[1] * ones(N),
    phase_κ[1] * ones(N),
    phase_c[1] * ones(N),
    adj,
)

const sixrd_lockdown_p = SixrdCountyLockdownParams(
    sixrd_p,
    phase_β,
    phase_μ,
    phase_α,
    phase_κ,
    phase_c,
    phase_max_distance,
    phase_compliance,
    phase_duration,
    population,
    counties,
    distances,
    Imax,
    grav_func,
    gm_p,
)

prob = DiscreteProblem(sixrd_county_lockdown!, u0, (0, 5), sixrd_lockdown_p)
sol = solve(prob, FunctionMap{true}(); callback=update_adj_cb)

