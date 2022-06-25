using IrelandCovidSim
using CovidSim

using GLMakie
using DataFrames
using JLD2
using InlineStrings
using Distances
using StatsBase
using LinearAlgebra
using GeoInterface
using GeoDataFrames
using GeometryBasics

import ArchGDAL as AG

########################################
# load data
########################################

function Base.convert(::Type{GeometryBasics.Polygon}, geom::AG.IGeometry{AG.wkbPolygon})
    coords = GeoInterface.coordinates(geom)

    pvec_ext = [GeometryBasics.Point(x...) for x in coords[1]]

    pvec_int = Vector{Vector{GeometryBasics.Point{2,Float64}}}()

    for i in coords[2:end]
        push!(pvec_int, [GeometryBasics.Point(x...) for x in i])
    end
    return GeometryBasics.Polygon(pvec_ext, pvec_int)
end

function Base.convert(
    ::Type{GeometryBasics.MultiPolygon}, geom::AG.IGeometry{AG.wkbMultiPolygon}
)
    poly_vec = Vector{GeometryBasics.Polygon{2,Float64}}()

    n = AG.ngeom(geom)

    for i in 0:(n - 1)
        push!(poly_vec, convert(GeometryBasics.Polygon, AG.getgeom(geom, i)))
    end
    return GeometryBasics.MultiPolygon(poly_vec)
end

ed_soa_df = GeoDataFrames.read(project_path("data", "processed", "ed_soa_shapefile"))

geo_poly_vec = Vector{GeometryBasics.Polygon}()
geo_multipoly_vec = Vector{GeometryBasics.MultiPolygon}()

geo_poly_pop = Vector{Float64}()
geo_multipoly_pop = Vector{Float64}()
geo_multipoly_area = Vector{Float64}()
geo_poly_area = Vector{Float64}()

for i in 1:nrow(ed_soa_df)
    geom = ed_soa_df.geom[i]
    n = ed_soa_df.population[i]
    if GeoInterface.geotype(geom) == :Polygon
        push!(geo_poly_vec, convert(GeometryBasics.Polygon, geom))
        push!(geo_poly_pop, n)
        push!(geo_poly_area, AG.geomarea(geom))
    elseif GeoInterface.geotype(geom) == :MultiPolygon
        push!(geo_multipoly_vec, convert(GeometryBasics.MultiPolygon, geom))
        push!(geo_multipoly_pop, n)
        push!(geo_multipoly_area, AG.geomarea(geom))
    end
end

fig = Figure()
ax = Axis(fig[1, 1]; aspect=AxisAspect(1))

poly_pop_per_area = geo_poly_pop ./ geo_poly_area ./ 1000^2
multipoly_pop_per_area = geo_multipoly_pop ./ geo_multipoly_area ./ 1000^2

cm = reverse(ColorSchemes.nuuk.colors)
cm = ColorSchemes.deep.colors

poly!(geo_poly_vec; color=log10.(poly_pop_per_area), colormap=cm)
poly!(geo_multipoly_vec; color=log10.(multipoly_pop_per_area), colormap=cm)

hidespines!(ax)
hidedecorations!(ax)

max_cb_val = max(maximum(poly_pop_per_area), maximum(multipoly_pop_per_area))
min_cb_val = max(minimum(poly_pop_per_area), minimum(multipoly_pop_per_area))

Colorbar(
    fig[1, 2];
    limits=((min_cb_val, max_cb_val)),
    colormap=cm,
    scale=log10,
    minorticksvisible=true,
)
Colorbar(
    fig[1, 2];
    limits=log10.((min_cb_val, max_cb_val)),
    colormap=cm,
    minorticksvisible=false,
    ticksvisible=false,
    labelvisible=false,
    ticklabelsvisible=false,
)

