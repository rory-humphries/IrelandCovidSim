using CovidSim
using CSV
using DataFrames
using JLD2

include(joinpath(@__DIR__, "..", "..", "src", "paths.jl"))

######################################
# Parameters
######################################

start_date = Date("2020-02-20")

######################################

df = CSV.read(joinpath(data_path(), "raw", "daily-covid-cases-deaths.csv"), DataFrame)
df = df[df.Entity .== "Ireland", :]
select!(df, Not([:Entity, :Code]))
rename!(
    df,
    [
        "Day" => :date,
        "Daily new confirmed deaths due to COVID-19" => :deaths,
        "Daily new confirmed cases of COVID-19" => :infections,
    ],
)

df_start_date = minimum(df.date)
df_end_date = maximum(df.date)
start_date_diff = Dates.value(df_start_date - start_date)
dates = start_date:Day(1):df_end_date

df = leftjoin(DataFrame(:date => dates), df; on=:date)
sort!(df, :date)

df[df[:, :deaths] .=== missing, :deaths] .= 0
df[df[:, :infections] .=== missing, :infections] .= 0

df.total_deaths = cumsum(df.deaths)

disallowmissing!(df)
save(joinpath(data_path(), "processed", "covid_data_from_$(start_date).jld2"), "df", df)
