using IrelandCovidSim
using CovidSim
using JLD2
using DataFrames
using CSV
using GeoDataFrames
using GeoFormatTypes

import ArchGDAL as AG

# have all datasets use this spatial projection
sp_ref = AG.importEPSG(29903)

########################################
# Northern Ireland Counties
########################################

ni_counties = GeoDataFrames.read(
    project_path("data", "raw", "Shapefiles", "northern_ireland_counties")
)

# remove unneeded cols
select!(ni_counties, ["geom", "CountyName"])
rename!(ni_counties, [:geom, :county])

AG.createcoordtrans(AG.getspatialref(ni_counties.geom[1]), sp_ref) do coordtrans
    for i in 1:nrow(ni_counties)
        AG.transform!(ni_counties.geom[i], coordtrans)
    end
end
########################################
# Republic of Ireland Counties
########################################

roi_counties = GeoDataFrames.read(project_path("data", "raw", "Shapefiles", "roi_counties"))
select!(roi_counties, ["geom", "COUNTY"])
rename!(roi_counties, [:geom, :county])

AG.createcoordtrans(AG.getspatialref(roi_counties.geom[1]), sp_ref) do coordtrans
    for i in 1:nrow(roi_counties)
        AG.transform!(roi_counties.geom[i], coordtrans)
    end
end

# combine cork city and council polygons
roi_counties[2, :geom] = AG.union(roi_counties[2, :geom], roi_counties[25, :geom])
deleteat!(roi_counties, 25)

# combine dublin city, south dublin, fingal and dun laoghaire polygons
roi_counties[1, :geom] = AG.union(roi_counties[1, :geom], roi_counties[10, :geom])
roi_counties[1, :geom] = AG.union(roi_counties[1, :geom], roi_counties[11, :geom])
roi_counties[1, :geom] = AG.union(roi_counties[1, :geom], roi_counties[22, :geom])
deleteat!(roi_counties, [10, 11, 22])

# combine galway city and council polygons
roi_counties[3, :geom] = AG.union(roi_counties[3, :geom], roi_counties[20, :geom])
deleteat!(roi_counties, 20)

########################################
# Combine Counties
########################################

ire_counties = vcat(ni_counties, roi_counties; cols=:intersect)
ire_counties.county .= lowercase.(ire_counties.county)

########################################
# NI super output areas (soa's)
########################################

# holds shapefile for northern irish super output areas
soa_df = GeoDataFrames.read(
    project_path("data", "raw", "Shapefiles", "super_output_areas", "SOA2011.shp")
)
rename!(soa_df, [:geom, :id, :name])
soa_df.id = lowercase.(soa_df.id)
soa_df.name = lowercase.(soa_df.name)

AG.createcoordtrans(AG.getspatialref(soa_df.geom[1]), sp_ref) do transform
    for i in 1:nrow(soa_df)
        AG.transform!(soa_df.geom[i], transform)
    end
end

########################################
# ROI electoral divisions (ed's)
########################################

ed_df = GeoDataFrames.read(project_path("data", "raw", "Shapefiles", "electoral_divisions"))
select!(ed_df, ["geom", "OSIED", "EDNAME"])
rename!(ed_df, ["geom", "id", "name"])
ed_df.name = lowercase.(ed_df.name)

AG.createcoordtrans(AG.getspatialref(soa_df.geom[1]), sp_ref) do transform
    for i in 1:nrow(ed_df)
        AG.transform!(ed_df.geom[i], transform)
    end
end

# find which osied's (id) are combined in the shapefile

# some ed's share a shapefile area
shared_ids = Dict{String,String}()
N = nrow(ed_df)
for i in 1:N
    #println(i)
    xsplit = split(ed_df[i, :id], '/')
    if length(xsplit) > 1
        ed_df[i, :id] = xsplit[1]
        df_row = copy(ed_df[i, :])
        push!(ed_df, df_row)
        ed_df[end, :id] = xsplit[2]
        shared_ids[string(parse(Int, xsplit[2]))] = string(parse(Int, xsplit[1]))
        if length(xsplit) > 2
            print("uh oh")
        end
    end
end
ed_df.id = string.(parse.(Int, ed_df.id))

########################################
# Combine ed's and soa's
########################################

ed_soa_df = vcat(soa_df, ed_df)

# find the counties each ed/soa belongs to
ed_soa_df.county .= ""

for i in 1:nrow(ire_counties)
    unknowns = ed_soa_df.county .== ""
    county_geom = ire_counties.geom[i]
    county = ire_counties.county[i]
    println(county, ", ", i)
    for j in (1:nrow(ed_soa_df))[unknowns]
        if AG.intersects(ed_soa_df[j, :geom], county_geom)
            ed_soa_df[j, :county] = county
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
ed_soa_pop.id = lowercase.(String[x[1] for x in id_name])
ed_soa_pop.name = lowercase.(String[x[2] for x in id_name])

ed_soa_df = leftjoin(ed_soa_df, ed_soa_pop[:, [:id, :Population]]; on=:id, makeunique=true)
rename!(ed_soa_df, lowercase.(names(ed_soa_df)))
ed_soa_df.population[ed_soa_df.population .=== missing] .= 0
deleteat!(ed_soa_df, ed_soa_df.population .== 0)

########################################
# Add controids
########################################

ed_soa_df.centroid_x .= 0.0
ed_soa_df.centroid_y .= 0.0

for i in 1:nrow(ed_soa_df)
    cent = AG.centroid(ed_soa_df[i, :geom])
    ed_soa_df[i, :centroid_x] = AG.getx(cent, 0)
    ed_soa_df[i, :centroid_y] = AG.gety(cent, 0)
end

########################################
# Commuting data
########################################

ed_travels_df = CSV.read(project_path("data", "raw", "ED_Used_Link_Info.csv"), DataFrame)
new_names = lowercase.(replace.(names(ed_travels_df), " " => "_"))
new_names[1] = "from_electoral_division"
new_names[2] = "from_county"
new_names[5] = "no_commuters"
rename!(ed_travels_df, new_names)

# only care about travels with given source/destination
deleteat!(ed_travels_df, ed_travels_df.to_electoral_division .== "No fixed place of work")
deleteat!(ed_travels_df, ed_travels_df.to_electoral_division .== "Work/school from home")

deleteat!(ed_travels_df, ed_travels_df.from_electoral_division .== "No fixed place of work")
deleteat!(ed_travels_df, ed_travels_df.from_electoral_division .== "Work/school from home")

ed_travels_df[
    :, [:from_electoral_division, :to_electoral_division, :from_county, :to_county]
] =
    lowercase.(
        ed_travels_df[
            :, [:from_electoral_division, :to_electoral_division, :from_county, :to_county]
        ],
    )

ed_travels_df.from_id .= [split(x, " - ")[1] for x in ed_travels_df.from_electoral_division]
ed_travels_df.to_id .= [split(x, " - ")[1] for x in ed_travels_df.to_electoral_division]

ed_travels_df = transform(
    groupby(ed_travels_df, :from_id), :no_commuters => sum => :out_commuters
)
ed_travels_df = transform(
    groupby(ed_travels_df, :to_id), :no_commuters => sum => :in_commuters
)

########################################
# Write data
########################################

crs = GeoFormatTypes.ProjString(AG.toPROJ4(sp_ref))

if !isdir(project_path("data", "processed", "ed_soa_shapefile"))
    mkdir(project_path("data", "processed", "ed_soa_shapefile"))
end
GeoDataFrames.write(
    project_path("data", "processed", "ed_soa_shapefile", "ed_soa.shp"),
    ed_soa_df;
    crs=crs,
)

save(project_path("data", "processed", "ed_travels_df.jld2"), "df", ed_travels_df)

ire_counties = leftjoin(
    ire_counties, combine(groupby(ed_soa_df, :county), :population => sum); on=:county
)

rename!(ire_counties, :population_sum=>:populaiton)
disallowmissing!(ire_counties)

if !isdir(project_path("data", "processed", "ire_counties_shapefile"))
    mkdir(project_path("data", "processed", "ire_counties_shapefile"))
end
GeoDataFrames.write(
    project_path("data", "processed", "ire_counties_shapefile", "ire_counties.shp"),
    ire_counties;
    crs=crs,
)

