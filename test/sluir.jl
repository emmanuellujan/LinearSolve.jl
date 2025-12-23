using LinearSolve, SparseArrays, LinearAlgebra, Test

@testset "SLUIRFactorization Integration" begin
    # Check if extension logic works (should stay loaded if SparseArrays is loaded)
    @info "Checking extension loading..."
    @test Base.get_extension(LinearSolve, :LinearSolveSLUIRExt) !== nothing

    n = 100
    A = sprand(n, n, 0.05) + I
    b = rand(n)
    prob = LinearProblem(A, b)
    
    @info "Solving with SLUIRFactorization..."
    sol = solve(prob, SLUIRFactorization())
    
    resid = norm(A * sol.u - b)
    @info "Residual: $resid"
    @test resid < 1e-8
end
