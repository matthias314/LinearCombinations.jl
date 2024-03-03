```@meta
DocTestSetup = doctestsetup
```

# Bases

## Basic functionality

```@docs
AbstractBasis
Basis
TensorBasis
getindex(::AbstractBasis{T,N}, ii::Vararg{Int,N}) where {T,N}
toindex
```

## `AbstractBasis` interface

A new subtype of `AbstractBasis` must provide a method for [`toindex`](@ref) and satisfy the
[abstract arrays interface](https://docs.julialang.org/en/v1/manual/interfaces/#man-interface-array)
for read-only arrays. The latter means that `size` and `getindex` must be supported.

As an example, here is subtype of `AbstractBasis` that turns ranges into bases.
The advantage over [`Basis`](@ref) is that no dictionary lookup is needed for `toindex`.
```julia
import Base: show, size, getindex
import LinearCombinations: toindex

struct RangeBasis{T,R<:AbstractRange{T}} <: AbstractBasis{T,1}
    range::R
end

show(io::IO, b::RangeBasis) = print(io, "RangeBasis(", b.range, ')')

size(b::RangeBasis) = size(b.range)

getindex(b::RangeBasis, i::Int) = b.range[i]

function toindex(b::RangeBasis, x)
    d, r = divrem(x-first(b.range), step(b.range))
    d += 1
    if r == 0 && firstindex(b.range) <= d <= lastindex(b.range)
        CartesianIndex(d)
    else
        error("$x is not an element of the basis $b")
    end
end
```
