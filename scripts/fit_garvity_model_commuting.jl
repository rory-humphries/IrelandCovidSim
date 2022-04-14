using CovidSim
using DataFrames
using InlineStrings
using Distances
using JLD2
using Optim

ed_soa_df = load(joinpath(data_path(), "processed", "ed_soa_df.jld2"), "df")
ed_travels_df = load(joinpath(data_path(), "processed", "ed_travels_df.jld2"), "df")


function exp_gravity(u, p)
    (α, β, γ) = p
    (attr_i, attr_j, dist) = u
    return attr_i^α * attr_j^β * exp(-γ * dist)
end

function exp_gravity(ui, uj, v, p)
    (α, β, γ) = p
    attr_i = ui[1]
    attr_j = uj[1]
    dist = v[1]
    return attr_i^α * attr_j^β * exp(-γ * dist)
end

struct GravityModel5{Ut,Vt,Ft,Pt} <: DiscreteMultivariateDistribution
    u::Ut
    v::Vt

    f::Ft

    p::Pt

    c::Float64
end

function GravityModel5(u, v, f, p)
    c = zeros(Threads.nthreads())
    N = size(u, 2)

    Threads.@threads for i in 1:N
        ui = u[:, i]
        for j in 1:N
            c[Threads.threadid()] += f(ui, u[:, j], v[i, j], p)
        end
    end
    return GravityModel5(u, v, f, p, sum(c))
end

Distributions.length(d::GravityModel5) = 2
Distributions.size(d::GravityModel5) = (2,)

function Distributions._logpdf(d::GravityModel5, x::AbstractVector)
    @unpack u, v, f, p, c = d

    return log(f(u[:, x[1]], u[:, x[2]], v[x[1], x[2]], p) / c)
end



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
travels = ed_travels_df.no_commuters
samples = [from_nid to_nid travels]

function op_func(p)
    gm = GravityModel5(u, v, exp_gravity, p)
    println(gm.c, ", ", p)
    if gm.c == Inf
        return Inf
    else
        vals = -sum(samples[:,3].*logpdf(gm, samples[:,1:2]'))
        println(vals)
        return vals
    end
end
opt = Optim.optimize(op_func, [0.0, 0.0, 0.0], [5.0, 5.0, 5.0], [0.1, 0.1, 1e-8];)

struct LazyMatrixMult{T1,T2,T3,At<:AbstractMatrix{T2},Bt<:AbstractMatrix{T2},Ft} <:
       AbstractMatrix{T3}
    A::At
    B::Bt
    f::Ft

    function LazyMatrixMult{At,Bt,Ft}(A::At, B::Bt, f::Ft=dot) where {At,Bt,Ft}
        return new{eltype(At),eltype(Bt),promote_type(eltype(At), eltype(Bt)),At,Bt,Ft}(
            A, B, f
        )
    end
end

Base.size(x::LazyMatrixMult) = (size(x.A, 1), size(x.B, 2))

Base.@propagate_inbounds function Base.getindex(x::LazyMatrixMult, I::Vararg{Int,2})
    return x.f(view(x.A, I[1], :), view(x.B, :, I[2]))
end

Base.@propagate_inbounds function Base.getindex(x::LazyMatrixMult, i::Int)
    sz = size(x)
    I = ((i - 1) % sz[1] + 1, floor(Int, (i - 1) / sz[1]) + 1)
    return getindex(x, I...)
end

function Base.sum(D::LazyMatrixMult)
    s = zeros(Threads.nthreads())
    Threads.@threads for i in 1:size(D, 2)
        tid = Threads.threadid()

        s[tid] += sum(D[:, i])
    end
    return sum(s)
end






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

