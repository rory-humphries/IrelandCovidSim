using GLMakie
using JLD2
using DataFrames
using Dates

start_date = Date("2020-02-20")
school_close_date = Date("2020-03-12")

stay_at_home = Date("2020-03-27")
phase_1 = Date("2020-05-18")
phase_2 = Date("2020-06-08")
phase_3 = Date("2020-06-29")
phase_4_paused = Date("2020-07-20")

phase_duration = [
    Dates.value(school_close_date - start_date),
    Dates.value(stay_at_home - school_close_date),
    Dates.value(phase_1 - stay_at_home),
    Dates.value(phase_2 - phase_1),
    Dates.value(phase_3 - phase_2),
    Dates.value(phase_4_paused - phase_3),
]
end_date = start_date + Day(sum(phase_duration))
dates = start_date:Day(1):phase_4_paused

df = load(joinpath(data_path(), "processed", "covid_data_from_$(start_date).jld2"), "df")

df.total_deaths_shifted = zeros(Int, length(dates))
df.total_deaths_shifted[1:(end - shift_deaths)] .= df.total_deaths[(shift_deaths + 1):end]
last_nz_id = findlast(!iszero, df.total_deaths_shifted)
df.total_deaths_shifted[last_nz_id:end] .= df.total_deaths_shifted[last_nz_id]

sim_df = load(
    joinpath(data_path(), "processed", "initial_lockdown_to_phase4_paused.jld2"), "df"
)

tick_step = 14
tick_locs = Int[]
tick_names = String[]
for i in eachindex(dates)[range(1, length(dates); step=tick_step)]
    push!(tick_locs, i)
    push!(tick_names, "$(monthabbr(dates[i])) $(day(dates[i]))")
end

lentime = length(dates)
slice_dates = range(1, lentime; step=lentime ÷ 7)

fig = Figure()
ax = Axis(
    fig[1, 1];
    xlabel="time",
    xticks=(tick_locs, tick_names),
    xticklabelrotation=π / 4,
    xticklabelalign=(:right, :center),
    ylabel="total deaths",
)

#ax.xticklabelrotation = π / 4
#ax.xticklabelalign = (:right, :center)

lines!(ax, sim_df.D)
vlines!(ax, cumsum(phase_duration) .+ 1)
lines!(ax, total_deaths_shifted[1:lentime])

