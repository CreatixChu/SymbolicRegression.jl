using Distributed
# addprocs(4)

@everywhere using Dates
@everywhere using DistributedArrays

# Initialize safely
A = fill("", 100)
A = distribute(A)

@sync @distributed for p in procs(A)
    local_A = localpart(A)
    for i in eachindex(local_A)
        local_A[i] = Dates.format(now(), "yyyymmdd_HHMMSS")
    end
end

println(collect(A))  # safer: gather to master before printing


# time JULIA_NUM_THREADS=4 julia parallel_example1.jl
#  4 thread: 3.302 s
#  1 thread: 10.332 s
# 10 thread: 1.319 s