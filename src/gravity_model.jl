struct GravityModel{Ut,Vt,Ft,Pt} <: DiscreteMultivariateDistribution
    u::Ut
    v::Vt

    f::Ft

    p::Pt

    c::Float64
end

function GravityModel(u, v, f, p)
    c = zeros(Threads.nthreads())
    N = size(u, 2)

    Threads.@threads for i in 1:N
        for j in 1:N
            c[Threads.threadid()] += f(u, v, p, i, j)
        end
    end
    return GravityModel(u, v, f, p, sum(c))
end

Distributions.length(d::GravityModel) = 2
Distributions.size(d::GravityModel) = (2,)

function Distributions._logpdf(d::GravityModel, x::AbstractVector)
    @unpack u, v, f, p, c = d

    return log(f(u, v, p, x[1], x[2]) / c)
end

struct GravityModelSampler <: Sampleable{Multivariate,Discrete}
    d::GravityModel
    rowsum::Vector{Float64} # probability of leaving from node i
    rowsumsum::Float64
end

function Distributions.sampler(d::GravityModel)
    N = size(d.u, 2)
    rowsum = zeros(N)

    Threads.@threads for i in 1:N
        rowsum[i] = sum(pdf(d, [[i, x] for x in 1:N]))
    end
    return GravityModelSampler(d, rowsum, sum(rowsum))
end

Base.length(s::GravityModelSampler) = 2

function Distributions._rand!(
    rng::AbstractRNG, s::GravityModelSampler, x::AbstractVector{T}
) where {T<:Integer}
    N = length(s.rowsum)
    row = sample(rng, Weights(s.rowsum, s.rowsumsum))
    col = sample(rng, Weights(pdf(s.d, [[row, x] for x in 1:N])))
    return x .= [row, col]
end

function Distributions._rand!(
    rng::AbstractRNG, s::GravityModelSampler, A::DenseMatrix{T}
) where {T<:Integer}
    N = length(s.rowsum)
    Nsamps = size(A, 2)

    row_samps = sample(rng, 1:N, Weights(s.rowsum, s.rowsumsum), Nsamps)

    k = 1
    for (i, count) in countmap(row_samps)
        col_samps = sample(rng, 1:N, Weights(pdf(s.d, [[i, x] for x in 1:N])), count)

        A[1, k:(k + count - 1)] .= i
        A[2, k:(k + count - 1)] .= col_samps
        k += count
    end
    return A
end

function Distributions.loglikelihood(d::GravityModel, samples, sample_counts)
    if d.c == Inf
        return Inf
    else
        return -sum(sample_counts .* logpdf(d, samples))
    end
end

function Distributions.fit_mle(::Type{GravityModel}, u, v, f, p0, samples, sample_counts)
    function op_func(p)
        d = GravityModel(u, v, f, p)
        return loglikelihood(d, samples, sample_counts)
    end
    opt = Optim.optimize(
        op_func, p0, NelderMead(), Optim.Options(; g_tol=1e-2, show_trace=true)
    )
    return GravityModel(u, v, f, Optim.minimizer(opt))
end

function exp_gravity(ui, uj, v, p)
    (α, β, γ) = p
    attr_i = ui[1]
    attr_j = uj[1]
    dist = v[1]
    return attr_i^α * attr_j^β * exp(-γ * dist)
end
