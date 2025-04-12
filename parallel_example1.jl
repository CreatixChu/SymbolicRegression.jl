using Base.Threads

function threaded_computation()
    results = zeros(10)
    @threads for i in 1:10
        sleep(1)
        results[i] = i^2
    end
    return results
end

println(Threads.nthreads())
rst = threaded_computation()
println(rst)

# time JULIA_NUM_THREADS=4 julia parallel_example1.jl
#  4 thread: 3.302 s
#  1 thread: 10.332 s
# 10 thread: 1.319 s