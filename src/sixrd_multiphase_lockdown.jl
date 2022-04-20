struct SixrdMultiphaseAffect{Sampt}
    # phase params
    βlist::Vector{Float64}
    μlist::Vector{Float64}
    αlist::Vector{Float64}
    κlist::Vector{Float64}
    clist::Vector{Float64}
    max_distance::Vector{Float64}
    compliance::Vector{Float64}
    phase_duration::Vector{Int64}

    # node and network properties
    cur_duration::Vector{Int64}
    cur_max_dist::Vector{Float64}
    cur_compliance::Vector{Float64}
    cur_phase::Vector{Int64}
    counties::Vector{String}
    Imax::Int64

    # sampler
    spl::Sampt
end
condition(t, u, integrator) = true

function (p::SixrdMultiphaseAffect)(integrator)
    N = length(p.counties)
    IJ = rand(p.spl, 10000)

    integrator.p[6] .= sparse(IJ[1, :], IJ[2, :], ones(10000), N, N)

    p.cur_duration .-= 1
    num_phases = size(p.phase_duration, 1)
    num_verts = size(p.cur_duration, 1)

    I_counties = accumulate_groups(integrator.u[:, 2], p.counties)

    for v in 1:num_verts

        # node belongs to county over Imax
        if I_counties[p.counties[v]] > p.Imax && p.cur_phase[v] > 3
            p.cur_phase[v] = 3
            p.cur_duration[v] = p.phase_duration[p.cur_phase[v]]
            # node reached end of duration in phase
        elseif p.cur_duration[v] == 0
            if p.cur_phase[v] <= num_phases
                p.cur_phase[v] += 1
                p.cur_duration[v] = p.phase_duration[p.cur_phase[v]]
            else
                p.cur_duration[v] = p.phase_duration[p.cur_phase[v]]
            end
        end
    end

    integrator.p[1] .= p.βlist[p.cur_phase]
    integrator.p[2] .= p.μlist[p.cur_phase]
    integrator.p[3] .= p.αlist[p.cur_phase]
    integrator.p[4] .= p.κlist[p.cur_phase]
    integrator.p[5] .= p.clist[p.cur_phase]
    return nothing
end

"""
    SixrdMultiphase(
        βlist::Vector{Float64}
        μlist::Vector{Float64}
        αlist::Vector{Float64}
        κlist::Vector{Float64}
        clist::Vector{Float64}
        max_distance::Vector{Float64}
        compliance::Vector{Float64}
        phase_duration::Vector{Int64}
        cur_duration::Vector{Int64}
        cur_max_dist::Vector{Float64}
        cur_compliance::Vector{Float64}
        cur_phase::Vector{Int64}
        counties::Vector{String}
        Imax::Int64
        spl::Sampt,
    ) where {Sampt}

A callback for the sixrd network model for simulating a multiphase lockdown strategy.

All nodes start in the final phase. Once the total infected in a county goes above Imax, all
nodes in the county go to phase one and their parameters are set accordingly. The nodes then
progress through the phases until they are back at the final (free) stage. .

- `βlist`: the ith element is the β associated with phse i
- `μlist`
- `αlist`
- `κlist`
- `clist`

"""
function SixrdMultiphase(args...; save_positions=(false, false), kwargs...)
    return DiscreteCallback(
        (t, u, integrator) -> true,
        SixrdMultiphaseAffect(args...);
        save_positions=save_positions,
        kwargs...,
    )
end

