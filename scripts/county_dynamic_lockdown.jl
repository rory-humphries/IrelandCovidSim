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
phase_duration = [50, 15, 52, 21, 21, 21, 21, 21, 400]
β_list = [0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37]
μ_list = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
c_list = [1.0, 0.7, 0.4, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75]
α_list = [0.0028, 0.0028, 0.0028, 0.0028, 0.0028, 0.0028, 0.0028, 0.0028, 0.0028]
κ_list = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
phase_max_distance = [1000.0, 2.0, 2.0, 2.0, 5.0, 5.0, 20.0, 1000.0, 1000.0]
phase_compliance = [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]
tmax = 200
Imax = 1400

seed_nodes = [250, 500, 1000]
seed_num = [4, 4, 4]

gm_p = (0.5212685415091278, 0.7486233175103572, 8.549615110124643e-5)
######################################
ed_soa_df = load(joinpath(data_path(), "processed", "ed_soa_df.jld2"), "df")


population = copy(ed_soa_df.population')
distances = zeros(nrow(ed_soa_df), nrow(ed_soa_df))
counties = ed_soa_df.county
for (i, (xi, yi)) in enumerate(zip(ed_soa_df.centroid_x, ed_soa_df.centroid_y))
    for (j, (xj, yj)) in enumerate(zip(ed_soa_df.centroid_x, ed_soa_df.centroid_y))
        distances[i, j] = haversine([xi, yi], [xj, yj])
    end
end


num_verts = size(population, 1)
num_phases = size(phase_duration, 1)

cur_duration = ones(Int64, num_verts) * phase_duration[1]
cur_max_dist = ones(Float64, num_verts) * phase_max_distance[1]
cur_compliance = ones(Float64, num_verts) * phase_compliance[1]
cur_phase = ones(Int64, num_verts)

gm = GravityModel(population, distances, exp_gravity, gm_p)
spl = sampler(gm)

cb = SixrdMultiphase(
    β_list,
    μ_list,
    α_list,
    κ_list,
    c_list,
    phase_max_distance,
    phase_compliance,
    phase_duration,
    cur_duration,
    cur_max_dist,
    cur_compliance,
    cur_phase,
    counties,
    Imax,
    spl,
)

N = length(population)
IJ = rand(spl, 10000)

adj = sparse(IJ[1, :], IJ[2, :], ones(10000), N, N)
p = (
    β_list[1] * ones(N),
    μ_list[1] * ones(N),
    α_list[1] * ones(N),
    κ_list[1] * ones(N),
    c_list[1] * ones(N),
    adj,
)

u0 = zeros(Float64, N, 5)
u0[:, 1] = population

for i in 1:size(seed_nodes, 1)
    u0[seed_nodes[i], 2] += seed_num[i]
    u0[seed_nodes[i], 1] -= seed_num[i]
end

prob = DiscreteProblem(sixrd!, u0, (0, tmax), p)
cb = DiscreteCallback(condition, aff; save_positions=(false, false))
sol = solve(prob, FunctionMap{true}(); callback=cb)

