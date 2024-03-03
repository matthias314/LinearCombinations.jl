using LinearCombinations
using StructEqualHash

StructEqualHash.typeid(::Type{Tensor}) = 0x1d5058d46de6d92f
StructEqualHash.typeid(::Type{AbstractTensor}) = 0x1d5058d46de6d92f

struct P{T} y::T end
@struct_equal_hash P
StructEqualHash.typeid(::Type{P}) = 0x4569ef9cc646f88f
