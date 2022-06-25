using CovidSim

using DataFrames
using JLD2
using InlineStrings
using Distances
using StatsBase
using Optim

import ArchGDAL as AG

include(joinpath(@__DIR__, "..", "..", "src", "paths.jl"))

########################################
# Parameters
########################################

p_init = [1.55, 0.032]
num_samples = 10000

########################################
# Load data
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

smpls = sample(1:nrow(ed_travels_df), num_samples)

function op_fun(p)
    α = p[1]
    β = p[2]

    println("p -> $p")

    M = sum(m .^ α)
    mlesum = 0.0
    for k in smpls
        i = ed_travels_df.from_nid[k]
        j = ed_travels_df.to_nid[k]

        dvec = d[i, :]

        s = radiation_si(dvec, n .^ β)
        pvec = (1 / (1 - (m[i]^α / M))) .* radiation_model(m[i]^α, n .^ β, s)
        p = pvec[j] / sum(pvec)
        if p != 0
            mlesum -= ed_travels_df.no_commuters[k]*log(p)
        end
    end
    println("mle -> $(mlesum)")
    return mlesum
end

inner_optimizer = LBFGS()
opt = Optim.optimize(
    p -> op_fun(p), [1.55, 1.05], inner_optimizer, Optim.Options(; show_trace=true)
)

