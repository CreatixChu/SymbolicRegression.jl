using Base.Threads
using Pkg
Pkg.activate(".")
import SymbolicRegression: SRRegressor
import MLJ: machine, fit!, predict, report


println("====== $(Threads.nthreads()) ======")

X = (a = rand(500), b = rand(500))
y = @. 2 * cos(X.a * 23.5) - X.b ^ 2
y = y .+ randn(500) .* 1e-3
model = SRRegressor(
    niterations=50,
    binary_operators=[+, -, *],
    unary_operators=[cos],
)
mach = machine(model, X, y)
@time begin
    fit!(mach)
end


# time JULIA_NUM_THREADS=4 julia parallel_example2.jl
#  4 thread: 37.55s user 1.91s system 191% cpu 20.643 total
#  1 thread: 26.85s user 1.41s system 101% cpu 27.755 total
# 10 thread: 56.50s user 1.89s system 302% cpu 19.276 total