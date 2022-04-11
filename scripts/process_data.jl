import ArchGDAL as AG
using DataFrames

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
#ni_soa = DataFrame(GeoTables.load("data/raw/Shapefiles/super_output_areas/SOA2011.shp"))
dataset = AG.read("data/raw/Shapefiles/super_output_areas/SOA2011.shp")
layer = AG.getlayer(dataset, 0)
source = AG.getspatialref(layer)
ni_soa = DataFrame(layer)
rename!(ni_soa, [:geometry, :id, :name])

AG.createcoordtrans(source, sp_ref) do transform
    for i in 1:nrow(ni_soa)
        AG.transform!(ni_soa[i,:geometry], transform)
    end
end

# Plotting
#
fig = Figure()
ax = Axis(fig[1, 1])
myvec = [wkb2geo(ni_soa[1,:geometry]

for i in 1:nrow(ni_soa)
    println(i)
    g = wkb2geo(ni_soa[i,:geometry])
    push!(myvec, g)
    #poly!(ax, wkb2geo(AG.simplifypreservetopology(ni_soa[i,1], 0.1)))
end

save("test.png", fig)

function wkb2geo(
    geom::AG.IGeometry{AG.wkbLineString},
    ::Val{N}=Val(AG.getcoorddim(geom)),
) where {N}
    n = AG.ngeom(geom)
    rvec = Vector{Point{N,Float32}}()
    for i in 0:(n - 1)
        push!(rvec, AG.getpoint(geom, i)[1:(N + 1)])
    end
    return LineString(rvec)
end

function wkb2geo(
    geom::AG.IGeometry{AG.wkbPolygon},
    ::Val{N}=Val(Int(AG.getcoorddim(geom))),
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
    geom::AG.IGeometry{AG.wkbMultiPolygon},
    ::Val{N}=Val(Int(AG.getcoorddim(geom))),
) where {N}
    npoly = AG.ngeom(geom)
    rvec = Vector{Polygon{N,Float32}}()
    for i in 0:(npoly - 1)
        push!(rvec, wkb2geo(AG.getgeom(geom, i), Val(N)))
    end
    return MultiPolygon(rvec)
end
