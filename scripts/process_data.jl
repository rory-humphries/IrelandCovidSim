using CovidSim
import ArchGDAL as AG
using JLD2
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
ire_counties.county .= lowercase.(ire_counties.county)

########################################
# NI super output areas (soa's)
########################################

# holds shapefile for northern irish super output areas
dataset = AG.read("data/raw/Shapefiles/super_output_areas/SOA2011.shp")
layer = AG.getlayer(dataset, 0)
source = AG.getspatialref(layer)
soa_df = DataFrame(layer)
rename!(soa_df, [:geometry, :id, :name])
soa_df.id = lowercase.(soa_df.id)
soa_df.name = lowercase.(soa_df.name)

AG.createcoordtrans(source, sp_ref) do transform
    for i in 1:nrow(soa_df)
        AG.transform!(soa_df[i, :geometry], transform)
    end
end

########################################
# ROI electoral divisions (ed's)
########################################

dataset = AG.read("data/raw/Shapefiles/electoral_divisions/electoral_divisions.shp")
layer = AG.getlayer(dataset, 0)
source = AG.getspatialref(layer)
ed_df = DataFrame(layer)
select!(ed_df, ["", "OSIED", "EDNAME"])
rename!(ed_df, ["geometry", "id", "name"])
ed_df.name = lowercase.(ed_df.name)

AG.createcoordtrans(source, sp_ref) do transform
    for i in 1:nrow(ed_df)
        AG.transform!(ed_df[i, :geometry], transform)
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
    county_geom = ire_counties.geometry[i]
    county = ire_counties.county[i]
    println(county, ", ", i)
    for j in (1:nrow(ed_soa_df))[unknowns]
        if AG.intersects(ed_soa_df[j, :geometry], county_geom)
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
ed_soa_pop.id = String[x[1] for x in id_name]
ed_soa_pop.name = String[x[2] for x in id_name]

ed_soa_df.population .= 0

ed_soa_df = leftjoin(ed_soa_df, ed_soa_pop[:, [:id, :Population]]; on=:id, makeunique=true)
ed_soa_df.Population[ed_soa_df.Population .=== missing] .= 0
ed_soa_df.population += ed_soa_df.Population
select!(ed_soa_df, Not(:Population))
deleteat!(ed_soa_df, ed_soa_df.population .== 0)

########################################
# Add controids
########################################

ed_soa_df.centroid_x .= 0.0
ed_soa_df.centroid_y .= 0.0

for i in 1:nrow(ed_soa_df)
    cent = AG.centroid(ed_soa_df[i, :geometry])
    ed_soa_df[i, :centroid_x] = AG.getx(cent, 0)
    ed_soa_df[i, :centroid_y] = AG.gety(cent, 0)
end

########################################
# Commuting data
########################################

ed_travels_df = CSV.read(joinpath(data_path(), "raw", "ED_Used_Link_Info.csv"), DataFrame)
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

save(joinpath(data_path(), "processed", "ed_soa_df.jld2"), "df", ed_soa_df)
save(joinpath(data_path(), "processed", "ed_travels_df.jld2"), "df", ed_travels_df)
