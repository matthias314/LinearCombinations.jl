using LinearCombinations

Base.hash(::Type{T}, h::UInt) where T = hash(string(T), h)  # for reproducible hashes
