#
# regrouping
#

export regroup, regroup_inv, Regroup, swap

function build_dict(ex, i, v, d)
    # @show ex i v
    if ex isa Expr
        ex.head === :tuple || error("malformed tree")
        for (j, a) in enumerate(ex.args)
            push!(v, j)
            i = build_dict(a, i, v, d)
            pop!(v)
        end
        i
    else
        haskey(d, ex) && error("malformed tree")
        d[ex] = i+1 => Expr(:call, :f, :x, v...)
        i+1
    end
end

function build_tree(ex, d, i, perm)
    if ex isa Expr
        ex.head === :tuple || error("malformed tree")
        Expr(:call, :g, (build_tree(a, d, i, perm) for a in ex.args)...)
    else
        haskey(d, ex) || error("incompatible trees")
        j = d[ex].first
        perm[j] == 0 || error("malformed tree")
        perm[j] = i[] += 1
        d[ex].second
    end
end

function tuple_from_expr(ex, d)
    if ex isa Expr
        ex.head === :tuple || error("malformed tree")
        Tuple(tuple_from_expr(a, d) for a in ex.args)
    else
        d[ex].first
    end
end

function regroup_tuples_data(expr_a, expr_b)
    d = Dict{Any,Pair{Int,Expr}}()
    n = build_dict(expr_a, 0, Int[], d)

    args = Vector{Expr}(undef, n)
    for (i, ex) in values(d)
        args[i] = ex
    end

    perm = zeros(Int, n)
    i = Ref(0)
    expr = build_tree(expr_b, d, i, perm)
    any(iszero, perm) && error("incompatible trees")

    aa = tuple_from_expr(expr_a, d)
    bb = tuple_from_expr(expr_b, d)

    aa, bb, (; expr, args, perm)
end

const RegroupCacheEltype = NamedTuple{(:expr, :args, :perm), Tuple{Expr, Vector{Expr}, Vector{Int}}}

const regroup_cache = IdDict{Any,RegroupCacheEltype}()

"""
    LinearCombinations.Regroup{A, B}

Applying a `Regroup` object to a Tensor or a linear combinations of tensors rearranges
the components of the tensor. Use `regroup` to create a `Regroup` object. It is possible
to define additional methods to apply `Regroup` objects to other arguments besides tensors.

See also [`regroup`](@ref).
"""
struct Regroup{A,B} end

# == is ===
# hash is computed from objectid

regroup_get(::Type{T}, field) where T <: Regroup = regroup_cache[T][field]

show(io::IO, rg::Regroup{A,B}) where {A,B} = print(io, "Regroup{$A,$B}")

"""
    regroup(a, b) -> Regroup

Return a `Regroup` object that can be used to rearrange the components of tensors and
possibly other structures.

The actual rearrangement is specified by the two parameters `a` and `b`.
Both are expression trees consisting of nested tuples of integers.
These trees encode the structure of nested tensors, and the integers specify
a mapping from the components of the nested source tensor to the nested target tensor.
The labels for `a` and `b` can in fact be of any `isbits` type instead of `Int`, but
they must be the same for `a` and `b`.

The return value `rg = regroup(a, b)` is a callable object. An argument `t` for `rg` must be
a nested tensor of the same shape as the `a` tree, and the return value is a `Tensor` of the same
shape as `b`. The components of the nested tensor `t` are permuted according to the labels.

If the components of `t` have non-zero degrees, then `rg(t)` additionally has a sign according to
the usual sign rule: whenever two ojects `x` and `y` are swapped, then this incurs
the sign `(-1)^(deg(x)*(deg(y)))`.

Moreover, `rg` is linear and can be called with linear combinations of tensors.

Note that for each `Regroup` element `rg`, Julia generates separate, efficient code for computing `rg(t)`.

See also [`swap`](@ref), [`regroup_inv`](@ref), [`Regroup`](@ref), [`$(@__MODULE__).DefaultCoefftype`](@ref).

# Examples

# Example without degrees

```jldoctest regroup
julia> rg = regroup(:( (1, (2, 3), 4) ), :( ((3, 1), (4, 2)) ))
Regroup{(1, (2, 3), 4),((3, 1), (4, 2))}

julia> t = Tensor("x", Tensor("y", "z"), "w")
x⊗(y⊗z)⊗w

julia> rg(t)
(z⊗x)⊗(w⊗y)
```

# Example with degrees

For simplicity, we define the degree of a `String` to be its length.
```jldoctest regroup
julia> LinearCombinations.deg(x::String) = length(x)

julia> rg(t)   # same rg and t as before
-(z⊗x)⊗(w⊗y)
```
"""
function regroup(a, b)
    aa, bb, data = regroup_tuples_data(a, b)
    aa isa Tuple || error("first expression must be a tuple")
    T = Regroup{aa,bb}
    regroup_cache[T] = data
    T()
end

deg(::Regroup) = Zero()

@linear rg::Regroup

keeps_filtered(::Regroup, ::Type) = true

"""
    regroup_inv(a, b) -> Tuple{Regroup,Regroup}

Return the tuple `(regroup(a, b), regroup(b, a))`.

See also [`regroup`](@ref).
"""
regroup_inv(a, b) = (regroup(a, b), regroup(b, a))

"""
    swap(t::AbstractTensor{Tuple{T1,T2}}) where {T1,T2} -> AbstractLinear{Tensor{Tuple{T2,T1}}}
    swap(a::AbstractLinear{AbstractTensor{Tuple{T1,T2}})}) where {T1,T2}
        -> AbstractLinear{Tensor{Tuple{T1,T2}})}

This linear function swaps the components of two-component tensors. If the two components
of a tensor `t` have non-zero degrees, then the usual sign `(-1)^(deg(t[1])*deg(t[2]))` is introduced.
By default, all terms have zero degree.

Note that `swap` is a special case of `regroup`:  it is simply defined as `regroup(:((1, 2)), :((2, 1)))`.

See also [`Tensor`](@ref), [`deg`](@ref), [`regroup`](@ref), [`$(@__MODULE__).DefaultCoefftype`](@ref).

# Examples

## Examples without degrees

```jldoctest swap
julia> t = Tensor("x", "z")
x⊗z

julia> swap(t)
z⊗x

julia> a = Linear("x" => 1, "yy" => 1) ⊗ Linear("z" => 1, "ww" => 1)
yy⊗ww+x⊗ww+x⊗z+yy⊗z

julia> swap(a)
ww⊗yy+z⊗x+z⊗yy+ww⊗x

julia> swap(a; coeff = 2)
2*ww⊗yy+2*z⊗x+2*z⊗yy+2*ww⊗x
```
## Examples with degrees

For simplicity, we define the degree of a `String` to be its length.
```jldoctest swap
julia> LinearCombinations.deg(x::String) = length(x)

julia> swap(t)   # same t as before
-z⊗x

julia> swap(a)   # same a as before
ww⊗yy-z⊗x+z⊗yy+ww⊗x
```
"""
const swap = regroup(:((1,2)), :((2,1)))

@propagate_inbounds @generated regroup_eval_expr(rg::Regroup, f, g, x) = regroup_get(rg, :expr)

#
# regrouping of tensors
#

_length(::Type{T}) where T <: Tuple = length(fieldtypes(T))
_length(::Type{<:AbstractTensor{T}}) where T <: Tuple = _length(T)

_getindex(x) = x
@propagate_inbounds _getindex(x, i) = x[i]
@propagate_inbounds _getindex(x, i, ii...) = _getindex(_getindex(x, i), ii...)

@propagate_inbounds _getindex(::Type{T}, i) where T <: Tuple = fieldtype(T, i)
@propagate_inbounds _getindex(::Type{<:AbstractTensor{T}}, i) where T <: Tuple = _getindex(T, i)

regroup_check_arg(::Type, ::Type, ::Type) = true

Base.@assume_effects :nothrow function regroup_check_arg(::Type{T}, ::Type{TT}, ::Type{TX}) where {T,TT<:Tuple,TX}
    n = _length(TT)
    TX <: T && _length(TX) == n &&
        all(ntuple(i -> regroup_check_arg(T, _getindex(TT, i), _getindex(TX, i)), n))
end

# @assume_effects allows to omit the sign computation for has_char2 coefficients
Base.@assume_effects :foldable :nothrow @generated function regroup_tensor_signexp(rg, f, x)
    args = Expr(:tuple, regroup_get(rg, :args)...)
    perm = regroup_get(rg, :perm)
    invdeg = (:(degs[$i]*degs[$j]) for j in 2:length(perm) for i in 1:j-1 if perm[i] > perm[j])
    quote
        degs = map(deg, $args)
        sum0(($(invdeg...),))
    end
end

@inline regroup_sign(rg, x, c) = signed(regroup_tensor_signexp(rg, _getindex, x), c)

@inline regroup_term(rg, x) = regroup_eval_expr(rg, _getindex, Tensor, x)

@linear_kw function (rg::Regroup{A,B})(x::T;
        coefftype = missing,
        addto = missing,
        coeff = one(DefaultCoefftype),
        is_filtered::Bool = false) where {A,B,T<:AbstractTensor}
    regroup_check_arg(AbstractTensor, typeof(A), T) ||
        error("argument type $T does not match first Regroup parameter $A")

    if addto !== missing
        R = _coefftype(addto)
    elseif coefftype !== missing
        R = unval(coefftype)
    else
        R = missing
    end

    if R === missing || !has_char2(R)
        coeff = regroup_sign(rg, x, coeff)
    end
    if R === missing
        R = typeof(coeff)
    end

    y = regroup_term(rg, x)

    if addto !== missing
        addmul!(addto, y, coeff; is_filtered)
    else
        Linear1{typeof(y),R}(y => coeff; is_filtered)
    end
end

#=
function return_type(rg::RG, ::Type{T}) where {RG<:Regroup,T<:AbstractTensor}
    R = return_type(regroup_sign, RG, T, DefaultCoefftype)
    U = return_type(regroup_term, RG, T)
    Linear1{U,R}
end
=#
