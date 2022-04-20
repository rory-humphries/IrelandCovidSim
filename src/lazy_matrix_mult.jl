struct LazyMatrixMult{T1,T2,T3,At<:AbstractMatrix{T2},Bt<:AbstractMatrix{T2},Ft} <:
       AbstractMatrix{T3}
    A::At
    B::Bt
    f::Ft

    function LazyMatrixMult{At,Bt,Ft}(A::At, B::Bt, f::Ft=dot) where {At,Bt,Ft}
        return new{eltype(At),eltype(Bt),promote_type(eltype(At), eltype(Bt)),At,Bt,Ft}(
            A, B, f
        )
    end
end

Base.size(x::LazyMatrixMult) = (size(x.A, 1), size(x.B, 2))

Base.@propagate_inbounds function Base.getindex(x::LazyMatrixMult, I::Vararg{Int,2})
    return x.f(view(x.A, I[1], :), view(x.B, :, I[2]))
end

Base.@propagate_inbounds function Base.getindex(x::LazyMatrixMult, i::Int)
    sz = size(x)
    I = ((i - 1) % sz[1] + 1, floor(Int, (i - 1) / sz[1]) + 1)
    return getindex(x, I...)
end

function Base.sum(D::LazyMatrixMult)
    s = zeros(Threads.nthreads())
    Threads.@threads for i in 1:size(D, 2)
        tid = Threads.threadid()

        s[tid] += sum(D[:, i])
    end
    return sum(s)
end


