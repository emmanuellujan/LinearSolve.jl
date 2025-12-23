module LinearSolveSLUIRExt

using LinearSolve
using SparseArrays
using LinearAlgebra

mutable struct SLUIRCache
    n::Int
    r64::Vector{Float64}
    work32::Vector{Float32}
    bf32::Vector{Float32}
    F32::Any
end

function LinearSolve.init_cacheval(
        alg::SLUIRFactorization, A, b, u, Pl, Pr, maxiters::Int, abstol, reltol,
        verbose::Union{Bool, LinearSolve.LinearVerbosity}, assump::LinearSolve.OperatorAssumptions)
    n = length(b)
    r64 = Vector{Float64}(undef, n)       # residual buffer (double)
    work32 = Vector{Float32}(undef, n)    # temp residual in single
    bf32 = Vector{Float32}(undef, n)      # temp right-hand side in single
    nz32 = Float32.(A.nzval)
    Af = SparseMatrixCSC{Float32, Int}(size(A,1), size(A,2),
                                        copy(A.colptr), copy(A.rowval),
                                        nz32)
    F32 = lu(Af)                          # single-precision sparse LU

    return SLUIRCache(n, r64, work32, bf32, F32)
end

function SciMLBase.solve!(cache::LinearSolve.LinearCache, alg::SLUIRFactorization; kwargs...)
    if cache.isfresh
        cache.cacheval.n = length(cache.b)
        cache.cacheval.r64 = Vector{Float64}(undef, cache.cacheval.n)       # residual buffer (double)
        cache.cacheval.work32 = Vector{Float32}(undef, cache.cacheval.n)    # temp residual in single
        cache.cacheval.bf32 = Vector{Float32}(undef, cache.cacheval.n)      # temp right-hand side in single
        nz32 = Float32.(cache.A.nzval)
        Af = SparseMatrixCSC{Float32, Int}(size(cache.A,1), size(cache.A,2),
                                           copy(cache.A.colptr), copy(cache.A.rowval),
                                           nz32)
        cache.cacheval.F32 = lu(Af)                                          # single-precision sparse LU

        cache.isfresh = false   
    end
    
    # Unpack from the mutable struct
    n = cache.cacheval.n
    r64 = cache.cacheval.r64
    work32 = cache.cacheval.work32
    bf32 = cache.cacheval.bf32
    F32 = cache.cacheval.F32
    
    b64 = eltype(cache.b) === Float64 ? cache.b : Vector{Float64}(cache.b)

    # Initial solve in single precision, accumulate in double
    @inbounds for i = 1:n
        bf32[i] = Float32(b64[i])
    end
    xf32 = F32 \ bf32
    x = Float64.(xf32)

    # Iterative refinement: compute residual in double, solve correction in single, update double solution
    for _ = 1:5
        mul!(r64, cache.A, x)                 # r64 = A * x (double)
        @inbounds for i = 1:n
            r64[i] = b64[i] - r64[i]          # r64 = b - A*x
            work32[i] = Float32(r64[i])       # convert residual to single
        end
        d32 = F32 \ work32
        @inbounds for i = 1:n
            x[i] += Float64(d32[i])           # update solution in double
        end
    end

    SciMLBase.build_linear_solution(alg, x, nothing, cache)
end

end