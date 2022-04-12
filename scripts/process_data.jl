import ArchGDAL as AG
using DataFrames
using CSV

########################################
# Northern Ireland Counties
########################################

dataset = AG.read("data/raw/Shapefiles/northern_ireland_counties")
layer = AG.getlayer(dataset, 0)

# have all datasets use this spatial projection
sp_ref = AG.getspatialref(layer)

ni_counties = DataFrame(layer)
# remove unneeded cols
select!(ni_counties, ["", "CountyName"])
rename!(ni_counties, [:geometry, :county])

########################################
# Republic of Ireland Counties
########################################

dataset = AG.read("data/raw/Shapefiles/roi_counties")
layer = AG.getlayer(dataset, 0)

roi_counties = DataFrame(layer)
select!(roi_counties, ["", "COUNTY"])
rename!(roi_counties, [:geometry, :county])

# combine cork city and council polygons
roi_counties[2, :geometry] = AG.union(
    roi_counties[2, :geometry], roi_counties[25, :geometry]
)
deleteat!(roi_counties, 25)

# combine dublin city, south dublin, fingal and dun laoghaire polygons
roi_counties[1, :geometry] = AG.union(
    roi_counties[1, :geometry], roi_counties[10, :geometry]
)
roi_counties[1, :geometry] = AG.union(
    roi_counties[1, :geometry], roi_counties[11, :geometry]
)
roi_counties[1, :geometry] = AG.union(
    roi_counties[1, :geometry], roi_counties[22, :geometry]
)
deleteat!(roi_counties, [10, 11, 22])

# combine galway city and council polygons
roi_counties[3, :geometry] = AG.union(
    roi_counties[3, :geometry], roi_counties[20, :geometry]
)
deleteat!(roi_counties, 20)

########################################
# Combine Counties
########################################

ire_counties = vcat(ni_counties, roi_counties; cols=:intersect)

########################################
# NI super output areas (soa's)
########################################

# holds shapefile for northern irish super output areas
dataset = AG.read("data/raw/Shapefiles/super_output_areas/SOA2011.shp")
layer = AG.getlayer(dataset, 0)
source = AG.getspatialref(layer)
ni_soa = DataFrame(layer)
rename!(ni_soa, [:geometry, :id1, :name])
ni_soa.id2 .= ""

AG.createcoordtrans(source, sp_ref) do transform
    for i in 1:nrow(ni_soa)
        AG.transform!(ni_soa[i, :geometry], transform)
    end
end

########################################
# ROI electoral divisions (ed's)
########################################

dataset = AG.read("data/raw/Shapefiles/electoral_divisions/electoral_divisions.shp")
layer = AG.getlayer(dataset, 0)
source = AG.getspatialref(layer)
roi_ed = DataFrame(layer)
select!(roi_ed, ["", "OSIED", "EDNAME"])
rename!(roi_ed, ["geometry", "id1", "name"])

AG.createcoordtrans(source, sp_ref) do transform
    for i in 1:nrow(roi_ed)
        AG.transform!(roi_ed[i, :geometry], transform)
    end
end

# find which osied's (id) are combined in the shapefile
roi_ed.id2 .= ""
for (i, x) in enumerate(roi_ed.id1)
    xsplit = split(x, '/')
    if length(xsplit) > 1
        roi_ed[i, :id1] = xsplit[1]
        roi_ed[i, :id2] = xsplit[2]
        if length(xsplit) > 2
            print("uh ih")
        end
    end
end
roi_ed.id1 = string.(parse.(Int, roi_ed.id1))
for i in 1:nrow(roi_ed)
    if roi_ed[i, :id2] != ""
        roi_ed[i, :id2] = string(parse(Int, roi_ed[i, :id2]))
    end
end

########################################
# Combine ed's and soa's
########################################

ire_ed_soa = vcat(ni_soa, roi_ed)

# find the counties each ed/soa belongs to
ire_ed_soa.county .= ""

for i in 1:nrow(ire_counties)
    unknowns = ire_ed_soa.county .== ""
    county_geom = ire_counties.geometry[i]
    county = ire_counties.county[i]
    println(county, ", ", i)
    for j in (1:nrow(ire_ed_soa))[unknowns]
        if AG.intersects(ire_ed_soa[j, :geometry], county_geom)
            ire_ed_soa[j, :county] = county
        end
    end
end

########################################
# Add populations
########################################

# hold info on every Electoral Division (ed) and super output area (soa)
ed_soa_pop = CSV.read("data/raw/Joined_Pop_Data_CSO_NISRA.csv", DataFrame)
select!(ed_soa_pop, ["Electoral Division", "Population"])

id_name = split.(ed_soa_pop[:, "Electoral Division"], " - ")
ed_soa_pop.id = String[x[1] for x in id_name]
ed_soa_pop.name = String[x[2] for x in id_name]

ire_ed_soa.population .= 0

ire_ed_soa = leftjoin(
    ire_ed_soa, ed_soa_pop[:, [:id, :Population]]; on=:id1 => :id, makeunique=true
)
ire_ed_soa.Population[ire_ed_soa.Population .=== missing] .= 0
ire_ed_soa.population += ire_ed_soa.Population
select!(ire_ed_soa, Not(:Population))

ire_ed_soa = leftjoin(
    ire_ed_soa, ed_soa_pop[:, [:id, :Population]]; on=:id2 => :id, makeunique=true
)
ire_ed_soa.Population[ire_ed_soa.Population .=== missing] .= 0
ire_ed_soa.population += ire_ed_soa.Population

select!(ire_ed_soa, Not(:Population))

########################################
# Add controids
########################################

ire_ed_soa.centroid_x .= 0.0
ire_ed_soa.centroid_y .= 0.0

for i in 1:nrow(ire_ed_soa)
    cent = AG.centroid(ire_ed_soa[i, :geometry])
    ire_ed_soa[i, :centroid_x] = AG.getx(cent, 0)
    ire_ed_soa[i, :centroid_y] = AG.gety(cent, 0)
end

"""
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
poly!(ax, myvec, color=1 ./ 1:length(myvec))


save("test.png", fig)
"""
