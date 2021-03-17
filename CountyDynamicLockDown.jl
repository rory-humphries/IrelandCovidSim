using DelimitedFiles
using StatsBase
using Random
using Distributions
using SparseArrays
using LinearAlgebra
using DynamicalSystems
using DifferentialEquations

include("SIXRD.jl")

######################################

# each duration is the number of days the phase lasts
duration_list   = [50,     15,     52,     21,     21,     21,     21,     21,     400]
β_list          = [0.37,   0.37,   0.37,   0.37,   0.37,   0.37,   0.37,   0.37 ,  0.37]
μ_list          = [0.1,    0.1,    0.1,    0.1,    0.1,    0.1,    0.1,    0.1,    0.1]
c_list          = [1.0,    0.7,    0.4,    0.4,    0.5,    0.6,    0.65,   0.7,    0.75]
α_list          = [0.0028, 0.0028, 0.0028, 0.0028, 0.0028, 0.0028, 0.0028, 0.0028, 0.0028]
κ_list          = [0.1,    0.1,    0.1,    0.1,    0.1,    0.1,    0.1,    0.1,    0.1]
max_dist_list   = [1000.0, 2.0,    2.0,    2.0,    5.0,    5.0,    20.0,   1000.0, 1000.0]
compliance_list = [0.7,    0.7,    0.7,    0.7,    0.7,    0.7,    0.7,    0.7,    0.7]
max_t           = 600

######################################

travel_probabilities = readdlm("/home/roryh/Repos/EpiGraph-cpp/data/processed/ed_soa_travel_prob_mat.csv", ',', Float64)
distances = readdlm("/home/roryh/Repos/EpiGraph-cpp/data/processed/ed_soa_dist_mat.csv", ',', Float64)
population = readdlm("/home/roryh/Repos/EpiGraph-cpp/data/processed/ed_soa_population.csv", ',', Int64, skipstart=1)[:,1]
counties = vec(readdlm("/home/roryh/Repos/EpiGraph-cpp/data/processed/ed_soa_county.csv", skipstart=1))

num_nodes   = size(population, 1)
num_phases  = size(duration_list, 1)
state       = zeros(Float64, num_nodes, 5)
state[:, 1] = population

p = SIXRDMetaPopParams()

p.params         = ones(Float64, num_nodes, 5)
p.params[:, 1] .*= β_list[1]
p.params[:, 2] .*= c_list[1]
p.params[:, 3] .*= μ_list[1]
p.params[:, 4] .*= α_list[1]
p.params[:, 5] .*= κ_list[1]

cur_duration   = ones(Int64, num_nodes) * duration_list[1]
cur_max_dist   = ones(Float64, num_nodes) * max_dist_list[1]
cur_compliance = ones(Float64, num_nodes) * compliance_list[1]
cur_phase = ones(Int64, num_nodes)

cat_vec = Array{Categorical,1}(undef, num_nodes)

for i = 1:num_nodes
    cat_vec[i] = Categorical(travel_probabilities[i, :])
end

state[3, 2] += 6

p.adj = gen_sparse_array(cat_vec, round.(Int64, population * 0.6))
prob = DiscreteProblem(SIXRDMetaPopODE, state, (0, 400), p) 
integrator = init(prob, FunctionMap())

# main sim loop
@time for t = 1:max_t

    step!(integrator)

    integrator.p.adj = gen_sparse_array(cat_vec, round.(Int64, population * 0.6))

    cur_duration .-= 1

    Threads.@threads for v = 1:num_nodes
        if cur_duration[v] == 0 && cur_phase[v] <= num_phases
            cur_phase[v] += 1
            cur_duration[v] = duration_list[cur_phase[v]]
        elseif cur_duration[v] == 0 && cur_phase[v] > num_phases
            cur_duration[v] = duration_list[cur_phase[v]]
        end
        
    end
    integrator.p.params[:, 1] = β_list[cur_phase]
    integrator.p.params[:, 2] = c_list[cur_phase]
    integrator.p.params[:, 3] = μ_list[cur_phase]
    integrator.p.params[:, 4] = α_list[cur_phase]
    integrator.p.params[:, 5] = κ_list[cur_phase]
end
