using CovidSim
using DataFrames
using Distributions
using InlineStrings
using Distances
using JLD2
using Optim

ed_soa_df = load(joinpath(data_path(), "processed", "ed_soa_df.jld2"), "df")
ed_travels_df = load(joinpath(data_path(), "processed", "ed_travels_df.jld2"), "df")

nd

u = ed_soa_df.population'
v = zeros(nrow(ed_soa_df), nrow(ed_soa_df))
for (i, (xi, yi)) in enumerate(zip(ed_soa_df.centroid_x, ed_soa_df.centroid_y))
    for (j, (xj, yj)) in enumerate(zip(ed_soa_df.centroid_x, ed_soa_df.centroid_y))
        v[i, j] = haversine([xi, yi], [xj, yj])
    end
end

ed_soa_df.nid .= 1:nrow(ed_soa_df)
from_nid = Int.(leftjoin(ed_travels_df, ed_soa_df[:, [:id, :nid]]; on=:from_id => :id).nid)
to_nid = Int.(leftjoin(ed_travels_df, ed_soa_df[:, [:id, :nid]]; on=:to_id => :id).nid)

sample_counts = ed_travels_df.no_commuters
samples = copy([from_nid to_nid]')

function mle_fit(::GravityModel, u, v, f, samples, sample_counts)
    return gravity_model_p(p) = GravityModel(u, v, f, p)
end

function op_func(p)
    gm = GravityModel(u, v, exp_gravity, p)
    if gm.c == Inf
        return Inf
    else
        vals = -sum(samples[:, 3] .* logpdf(gm, samples[:, 1:2]'))
        return vals
    end
end
opt = Optim.optimize(
    op_func, [0.1, 0.1, 1e-8], NelderMead(), Optim.Options(; g_tol=1e-3, show_trace=true)
)
##
# Regression
ed_travels_df = leftjoin(
    ed_travels_df,
    ed_soa_df[:, [:id, :population]];
    on=:from_id => :id,
    renamecols="" => "_out",
)
ed_travels_df = leftjoin(
    ed_travels_df,
    ed_soa_df[:, [:id, :population]];
    on=:to_id => :id,
    renamecols="" => "_in",
)

leftjoin(ed_travels_df, ed_soa_df[:, [:id, :population]]; on=:from_id => :id)
tab = DataFrame(;
    y=log.(ed_travels_df.no_commuters),
    x1=log.(ed_travels_df.population_out),
    x2=log.(ed_travels_df.population_in),
    x3=ed_travels_df.distance,
)
exp.(predict(xm)) .- ed_travels_df.no_commuters

# Plotting
#
using CairoMakie
using GeometryBasics

function wkb2geo(
    geom::AG.IGeometry{AG.wkbLineString}, ::Val{N}=Val(AG.getcoorddim(geom))
) where {N}
    n = AG.ngeom(geom)
    rvec = Vector{Point{N,Float32}}()
    for i in 0:(n - 1)
        push!(rvec, AG.getpoint(geom, i)[1:(N + 1)])
    end
    return LineString(rvec)
end

function wkb2geo(
    geom::AG.IGeometry{AG.wkbPolygon}, ::Val{N}=Val(Int(AG.getcoorddim(geom)))
) where {N}
    nline = AG.ngeom(geom)
    rvec = Vector{Vector{Point{N,Float32}}}([[] for _ in 1:nline])
    for i in 0:(nline - 1)
        g = AG.getgeom(geom, i)
        npoint = AG.ngeom(g)
        for j in 0:(npoint - 1)
            push!(rvec[i + 1], AG.getpoint(g, j)[1:N])
        end
    end
    return Polygon(rvec[1], rvec[2:end])
end

function wkb2geo(
    geom::AG.IGeometry{AG.wkbMultiPolygon}, ::Val{N}=Val(Int(AG.getcoorddim(geom)))
) where {N}
    npoly = AG.ngeom(geom)
    rvec = Vector{Polygon{N,Float32}}()
    for i in 0:(npoly - 1)
        push!(rvec, wkb2geo(AG.getgeom(geom, i), Val(N)))
    end
    return MultiPolygon(rvec)
end

fig = Figure()
ax = Axis(fig[1, 1])
myvec = MultiPolygon[]

for i in 1:nrow(ire_ed_soa)
    println(i)
    g = wkb2geo(ire_ed_soa[i, :geometry])
    if g isa Polygon
        push!(myvec, MultiPolygon([g]))
    else
        push!(myvec, g)
    end
    #poly!(ax, g)
end
poly!(ax, myvec; color=log10.(ed_soa_df.population))

save("test.png", fig)

