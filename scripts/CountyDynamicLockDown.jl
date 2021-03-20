using DelimitedFiles
using StatsBase
using Random
using Distributions
using SparseArrays
using LinearAlgebra
using DifferentialEquations
using Plots

using CovidSim

data_path = normpath(joinpath(@__DIR__, "..", "data"))
# Parameters
######################################

# each duration is the number of days the phase lasts
phase_duration     = [50,     15,     52,     21,     21,     21,     21,     21,     400]
β_list             = [0.37,   0.37,   0.37,   0.37,   0.37,   0.37,   0.37,   0.37 ,  0.37]
μ_list             = [0.1,    0.1,    0.1,    0.1,    0.1,    0.1,    0.1,    0.1,    0.1]
c_list             = [1.0,    0.7,    0.4,    0.4,    0.5,    0.6,    0.65,   0.7,    0.75]
α_list             = [0.0028, 0.0028, 0.0028, 0.0028, 0.0028, 0.0028, 0.0028, 0.0028, 0.0028]
κ_list             = [0.1,    0.1,    0.1,    0.1,    0.1,    0.1,    0.1,    0.1,    0.1]
phase_max_distance = [1000.0, 2.0,    2.0,    2.0,    5.0,    5.0,    20.0,   1000.0, 1000.0]
phase_compliance   = [0.7,    0.7,    0.7,    0.7,    0.7,    0.7,    0.7,    0.7,    0.7]
tmax               = 20
Imax               = 1400
######################################

# add population to sim struct


travel_probabilities = readdlm(joinpath(data_path, "processed", "ed_soa_travel_prob_mat.csv"), ',', Float64)
distances = readdlm(joinpath(data_path, "processed", "ed_soa_dist_mat.csv"), ',', Float64)
population = readdlm(joinpath(data_path, "processed", "ed_soa_population.csv"), ',', Int64, skipstart=1)[:,1]
counties = vec(readdlm(joinpath(data_path, "processed", "ed_soa_county.csv"), String, skipstart=1))

struct Phases
    β::Vector{Float64}
    c::Vector{Float64}
    μ::Vector{Float64}
    α::Vector{Float64}
    κ::Vector{Float64}

    max_distance::Vector{Float64}
    compliance::Vector{Float64}
    duration::Vector{Int64}
end

struct Nodes
    cur_duration::Vector{Int64}
    cur_max_dist::Vector{Float64}
    cur_compliance::Vector{Float64}
    cur_phase::Vector{Int64}
    cat_vec::Vector{Categorical}
    counties::Vector{String}
end

mutable struct Sim
    # phase params
    phases::Phases

    # node and network properties
    nodes::Nodes
    Imax::Int64

    # sixrd params
    p::Matrix{Float64}
    adj::SparseMatrixCSC{Float64}
end

function SimOde(state::Matrix{Float64}, sim::Sim, t)
    return SIXRDMetaPopODE(state, sim.p, sim.adj)
end

condition = function (t, u, integrator)
    true
end

function update_sim!(sim::Sim, state::Matrix{Float64})
    sim.adj = gen_sparse_array(sim.nodes.cat_vec, 
    round.(Int64, population * 0.6))
    
    sim.nodes.cur_duration .-= 1
    num_phases = size(sim.phases.duration, 1)
    num_verts = size(sim.nodes.cur_duration, 1)
    
    I_counties = accumulate_groups(state[:, 2], sim.nodes.counties)
    
    @inbounds Threads.@threads for v = 1:num_verts

        # node belongs to county over Imax
        if I_counties[sim.nodes.counties[v]] > Imax && sim.nodes.cur_phase[v] > 3
            sim.nodes.cur_phase[v] = 3
            sim.nodes.cur_duration[v] = sim.phases.duration[sim.nodes.cur_phase[v]]
        # node reached end of duration in phase
        elseif sim.nodes.cur_duration[v] == 0 
            if sim.nodes.cur_phase[v] <= num_phases
                sim.nodes.cur_phase[v] += 1
                sim.nodes.cur_duration[v] = sim.phases.duration[sim.nodes.cur_phase[v]]
            else 
                sim.nodes.cur_duration[v] = sim.phases.duration[sim.nodes.cur_phase[v]]
            end
        end
    end
    
    sim.p[:, 1] = sim.phases.β[sim.nodes.cur_phase]
    sim.p[:, 2] = sim.phases.c[sim.nodes.cur_phase]
    sim.p[:, 3] = sim.phases.μ[sim.nodes.cur_phase]
    sim.p[:, 4] = sim.phases.α[sim.nodes.cur_phase]
    sim.p[:, 5] = sim.phases.κ[sim.nodes.cur_phase]
end

function f!(integrator)
    update_sim!(integrator.p, integrator.u)
end


num_verts   = size(population, 1)
num_phases  = size(phase_duration, 1)

phases = Phases(β_list, c_list, μ_list, α_list, κ_list, 
phase_max_distance, phase_compliance, phase_duration)

cur_duration   = ones(Int64, num_verts) * phase_duration[1]
cur_max_dist   = ones(Float64, num_verts) * phase_max_distance[1]
cur_compliance = ones(Float64, num_verts) * phase_compliance[1]
cur_phase      = ones(Int64, num_verts)

cat_vec = Array{Categorical,1}(undef, num_verts)

for i = 1:num_verts
    cat_vec[i] = Categorical(travel_probabilities[i, :])
end

nodes = Nodes(cur_duration, cur_max_dist, cur_compliance, 
cur_phase, cat_vec, counties)


p = zeros(Float64, num_verts, 5)
p[:, 1] .*= β_list[1]
p[:, 2] .*= c_list[1]
p[:, 3] .*= μ_list[1]
p[:, 4] .*= α_list[1]
p[:, 5] .*= κ_list[1]

state       = zeros(Float64, num_verts, 5)
state[:, 1] = population

adj = gen_sparse_array(cat_vec, round.(Int64, population * 0.6))

sim = Sim(phases, nodes, Imax, p, adj)

state[3, 2] += 6

prob = DiscreteProblem(SimOde, state, (0, tmax), sim) 
cb = DiscreteCallback(condition, f!, save_positions=(true, false))
sol = solve(prob, callback=cb)

