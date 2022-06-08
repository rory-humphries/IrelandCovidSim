struct SixrdNetworkCountyLockdownParams{Pt,Sampt}
    # sixrd params
    sixrd_p::Pt

    # phase params
    phase_β::Vector{Float64}
    phase_μ::Vector{Float64}
    phase_α::Vector{Float64}
    phase_κ::Vector{Float64}
    phase_c::Vector{Float64}
    phase_max_distance::Vector{Float64}
    phase_compliance::Vector{Float64}
    phase_duration::Vector{Int64}

    # node and network properties
    counties::Vector{String}
    cur_duration::Vector{Int64}
    cur_max_dist::Vector{Float64}
    cur_compliance::Vector{Float64}
    cur_phase::Vector{Int64}

    # others
    Imax::Int64
    spl::Sampt # sampler
end

function SixrdNetworkCountyLockdownParams(
    sixrd_p::Pt,
    phase_β::Vector{Float64},
    phase_μ::Vector{Float64},
    phase_α::Vector{Float64},
    phase_κ::Vector{Float64},
    phase_c::Vector{Float64},
    phase_max_distance::Vector{Float64},
    phase_compliance::Vector{Float64},
    phase_duration::Vector{Int64},
    counties::Vector{String},
    Imax::Int64,
    spl::Sampt,
) where {Pt,Sampt}
    Nv = length(counties)
    cur_duration = ones(Int64, Nv) * phase_duration[1]
    cur_max_dist = ones(Float64, Nv) * phase_max_distance[1]
    cur_compliance = ones(Float64, Nv) * phase_compliance[1]
    cur_phase = ones(Int64, Nv)

    return SixrdNetworkCountyLockdownParams(
        sixrd_p,
        phase_β,
        phase_μ,
        phase_α,
        phase_κ,
        phase_c,
        phase_max_distance,
        phase_compliance,
        phase_duration,
        counties,
        cur_duration,
        cur_max_dist,
        cur_compliance,
        cur_phase,
        Imax,
        spl,
    )
end

function affect_params1!(integrator)
    #Nv = size(integrator.u, 1)
    Np = size(integrator.p.phase_duration, 1)
    # IJ = rand(cb.spl, 10000)

    #integrator.p.sixrd_p[6] .= sparse(IJ[1, :], IJ[2, :], ones(10000), Nv, Nv)
    integrator.p.cur_duration .-= 1

    I_counties = accumulate_groups(integrator.u[:, 2], integrator.p.counties)

    for v in 1:num_verts

        # node belongs to county over Imax
        if I_counties[integrator.p.counties[v]] > cb.Imax && integrator.p.cur_phase[v] > 3
            integrator.p.cur_phase[v] = 3
            integrator.p.cur_duration[v] = cb.phase_duration[integrator.p.cur_phase[v]]
            # node reached end of duration in phase
        elseif integrator.p.cur_duration[v] == 0
            if integrator.p.cur_phase[v] <= Np
                integrator.p.cur_phase[v] += 1
                integrator.p.cur_duration[v] = integrator.p.phase_duration[integrator.p.cur_phase[v]]
            else
                integrator.p.cur_duration[v] = integrator.p.phase_duration[integrator.p.cur_phase[v]]
            end
        end
    end

    integrator.p.sixrd_p[1] .= integrator.p.phase_β[integrator.p.cur_phase]
    integrator.p.sixrd_p[2] .= integrator.p.phase_μ[integrator.p.cur_phase]
    integrator.p.sixrd_p[3] .= integrator.p.phase_α[integrator.p.cur_phase]
    integrator.p.sixrd_p[4] .= integrator.p.phase_κ[integrator.p.cur_phase]
    integrator.p.sixrd_p[5] .= integrator.p.phase_c[integrator.p.cur_phase]
    return nothing
end

struct SixrdMultiphaseAffect{Sampt}
    # phase params
    phase_β::Vector{Float64}
    phase_μ::Vector{Float64}
    phase_α::Vector{Float64}
    phase_κ::Vector{Float64}
    phase_c::Vector{Float64}
    phase_max_distance::Vector{Float64}
    phase_compliance::Vector{Float64}
    phase_duration::Vector{Int64}

    # node and network properties
    counties::Vector{String}
    cur_duration::Vector{Int64}
    cur_max_dist::Vector{Float64}
    cur_compliance::Vector{Float64}
    cur_phase::Vector{Int64}

    # others
    Imax::Int64
    spl::Sampt # sampler
end

function (cb::SixrdMultiphaseAffect)(integrator)
    N = length(cb.counties)
    IJ = rand(cb.spl, 10000)

    integrator.p[6] .= sparse(IJ[1, :], IJ[2, :], ones(10000), N, N)

    cb.cur_duration .-= 1
    num_phases = size(cb.phase_duration, 1)
    num_verts = size(cb.cur_duration, 1)

    I_counties = accumulate_groups(integrator.u[:, 2], cb.counties)

    for v in 1:num_verts

        # node belongs to county over Imax
        if I_counties[cb.counties[v]] > cb.Imax && cb.cur_phase[v] > 3
            cb.cur_phase[v] = 3
            cb.cur_duration[v] = cb.phase_duration[cb.cur_phase[v]]
            # node reached end of duration in phase
        elseif cb.cur_duration[v] == 0
            if cb.cur_phase[v] <= num_phases
                cb.cur_phase[v] += 1
                cb.cur_duration[v] = cb.phase_duration[cb.cur_phase[v]]
            else
                cb.cur_duration[v] = cb.phase_duration[cb.cur_phase[v]]
            end
        end
    end

    integrator.p[1] .= cb.βlist[cb.cur_phase]
    integrator.p[2] .= cb.μlist[cb.cur_phase]
    integrator.p[3] .= cb.αlist[cb.cur_phase]
    integrator.p[4] .= cb.κlist[cb.cur_phase]
    integrator.p[5] .= cb.clist[cb.cur_phase]
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
function SixrdMultiphase(
    phase_β::Vector{Float64},
    phase_μ::Vector{Float64},
    phase_α::Vector{Float64},
    phase_κ::Vector{Float64},
    phase_c::Vector{Float64},
    phase_max_distance::Vector{Float64},
    phase_compliance::Vector{Float64},
    phase_duration::Vector{Int64},
    counties::Vector{String},
    Imax::Int64,
    spl::Sampt;
    save_positions=(false, false),
    kwargs...,
)
    num_verts = length(counties)
    cur_duration = ones(Int64, num_verts) * phase_duration[1]
    cur_max_dist = ones(Float64, num_verts) * phase_max_distance[1]
    cur_compliance = ones(Float64, num_verts) * phase_compliance[1]
    cur_phase = ones(Int64, num_verts)

    return DiscreteCallback(
        (t, u, integrator) -> true,
        SixrdMultiphaseAffect(
            phase_β::Vector{Float64},
            phase_μ::Vector{Float64},
            phase_α::Vector{Float64},
            phase_κ::Vector{Float64},
            phase_c::Vector{Float64},
            phase_max_distance::Vector{Float64},
            phase_compliance::Vector{Float64},
            phase_duration::Vector{Int64},
            cur_duration::Vector{Int64},
            cur_max_dist::Vector{Float64},
            cur_compliance::Vector{Float64},
            cur_phase::Vector{Int64},
            counties::Vector{String},
            Imax::Int64,
            spl::Sampt,
        );
        save_positions=save_positions,
        kwargs...,
    )
end

