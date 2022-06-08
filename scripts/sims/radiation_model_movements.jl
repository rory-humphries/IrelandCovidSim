using CovidSim
using DataFrames
using JLD2
using InlineStrings
using Distances
using StatsBase

import ArchGDAL as AG

include(joinpath(@__DIR__, "..", "..", "src", "paths.jl"))

########################################
# parameters and data
########################################

α = 1.463
β = 0.757
N = 5 * 10^4

write_data = true
########################################
# load data
########################################

ed_soa_df = load(joinpath(data_path(), "processed", "ed_soa_df.jld2"), "df")
ed_travels_df = load(joinpath(data_path(), "processed", "ed_travels_df.jld2"), "df")

# remove commuting with distance 0
delete!(ed_travels_df, ed_travels_df.distance .== 0)

ed_soa_df.nid .= 1:nrow(ed_soa_df)
ed_travels_df = leftjoin(ed_travels_df, ed_soa_df[:, [:id, :nid]]; on=:from_id => :id)
rename!(ed_travels_df, :nid => :from_nid)
ed_travels_df = leftjoin(ed_travels_df, ed_soa_df[:, [:id, :nid]]; on=:to_id => :id)
rename!(ed_travels_df, :nid => :to_nid)

od_df = combine(groupby(ed_travels_df, [:from_id]), :no_commuters => sum)
leftjoin!(ed_soa_df, od_df; on=:id => :from_id)
rename!(ed_soa_df, :no_commuters_sum => :out_degree)

# only allow movement to roi to better compare distances
ed_soa_df.population[ed_soa_df.out_degree .=== missing] .= 0
ed_soa_df.out_degree[ed_soa_df.out_degree .=== missing] .= 0

dropmissing!(ed_soa_df)

m = ed_soa_df.population
n = m
d = zeros(nrow(ed_soa_df), nrow(ed_soa_df))
for (i, (xi, yi)) in enumerate(zip(ed_soa_df.centroid_x, ed_soa_df.centroid_y))
    for (j, (xj, yj)) in enumerate(zip(ed_soa_df.centroid_x, ed_soa_df.centroid_y))
        d[i, j] = haversine([xi, yi], [xj, yj])
    end
end

########################################
# sim trades
########################################

i_samps = sample(1:nrow(ed_soa_df), Weights(ed_soa_df.out_degree), N)
j_samps = zeros(Int, N)

M = sum(m .^ α)
for ind in 1:length(i_samps)
    i = i_samps[ind]

    dvec = d[i, :]

    s = radiation_si(dvec, n .^ β)
    pvec = (1 / (1 - (m[i]^α / M))) .* radiation_model(m[i]^α, n .^ β, s)
    if sum(pvec) == 0 
        continue 
    end
    j = sample(1:nrow(ed_soa_df), Weights(pvec))
    j_samps[ind] = j
end

samp_dists = [d[i,j] for (i,j) in zip(i_samps, j_samps)]
nz_dists = samp_dists .> 0
samp_dists = samp_dists[nz_dists]

sample_df = DataFrame(
    :source => ed_soa_df[i_samps[nz_dists], :id],
    :destination => ed_soa_df[j_samps[nz_dists], :id],
    :distance => samp_dists,
)

########################################
# write data
########################################

if write_data
    save(
        joinpath(
            data_path(),
            "processed",
            "radiation_model_sims",
            "roi_α=$(α)_β=$(β)_N=$(N).jld2",
        ),
        "df",
        sample_df,
    )
end

