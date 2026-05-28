#
# regrouping
#

export regroup, regroup_inv, @regroup_str, @regroup_inv_str, Regroup, swap

"""
    $(@__MODULE__).Regroup{A, B}

Applying a `Regroup` object to a Tensor or a linear combinations of tensors rearranges
the components of the tensor. Use the `regroup""` string macro to create a `Regroup` object.
It is possible to define additional methods to apply `Regroup` objects to other arguments
besides tensors.

See also [`@regroup_str`](@ref).
"""
struct Regroup{A,B} end

# == is ===
# hash is computed from objectid

show(io::IO, rg::Regroup{A,B}) where {A,B} = print(io, "Regroup{$A, $B}")

# type parameters in method signatures disable @nospecialize
regroup_a(::Regroup{A}) where A = A
regroup_b(::Regroup{A,B}) where {A,B} = B

function regroup_build!((d, i), ex::Expr)
    ex.head == :tuple || error("malformed source tree")
    Tuple(map(Fix1(regroup_build!, (d, i)), ex.args))
end

function regroup_build!((d, i), ex)
    haskey(d, ex) ? error("malformed source tree") : d[ex] = i[] += 1
end

function regroup_replace!(d, ex::Expr)
    ex.head == :tuple || error("malformed target tree")
    Tuple(map(Fix1(regroup_replace!, d), ex.args))
end

function regroup_replace!(d, ex)
    haskey(d, ex) ? pop!(d, ex) : error("malformed or incompatible target tree")
end

"""
    regroup"a -> b" -> Regroup

Create a `Regroup` object that can be used to rearrange the components of tensors and
possibly other structures.

The actual rearrangement is specified by the two parameters `a` and `b`,
which are (possibly nested) tuples of integers.
These tuples encode the structure of nested tensors, and the integers specify
a mapping from the components of the nested source tensor to the nested target tensor.
The labels for `a` and `b` can in fact be of any `isbits` type or `Symbol` instead of
`Int`, but they must be the same for `a` and `b`.

The created object `rg = regroup"a -> b"` is callable. An argument `t` for `rg` must be
a nested tensor of the same shape as the `a` tree, and the return value is a `Tensor` of the same
shape as `b`. The components of the nested tensor `t` are permuted according to the labels.

If the components of `t` have non-zero degrees, then `rg(t)` additionally has a sign according to
the usual sign rule: whenever two ojects `x` and `y` are swapped, then this incurs
the sign `(-1)^(deg(x)*(deg(y)))`.
In this case the returned value is of type `Linear1` instead of `Tensor`.

Moreover, `rg` is linear and can be called with linear combinations of tensors.

Note that for each `Regroup` element `rg`, Julia generates separate, efficient code for computing `rg(t)`.

See also [`swap`](@ref), [`@regroup_inv_str`](@ref), [`Regroup`](@ref), [`$(@__MODULE__).DefaultCoefftype`](@ref).

# Examples

# Example without degrees

```jldoctest regroup
julia> rg = regroup"(1, (2, 3), 4) -> ((3, 1), (4, 2))"
Regroup{(1, (2, 3), 4), ((3, 1), (4, 2))}

julia> rg == regroup"(a, (b, c), d) -> ((c, a), (d, b))"
true

julia> t = Tensor("x", Tensor("y", "z"), "w")
"x"⊗("y"⊗"z")⊗"w"

julia> rg(t)
("z"⊗"x")⊗("w"⊗"y")
```

# Example with degrees

The degree of a `GradedString` (created with the `gr""` string macro) is its length.
```jldoctest regroup
julia> using $(@__MODULE__).TestHelpers: GradedString, @gr_str

julia> t = Tensor(gr"x", Tensor(gr"y", gr"z"), gr"w")
gr"x"⊗(gr"y"⊗gr"z")⊗gr"w"

julia> rg(t)   # same rg as before
Linear1{Tensor{Tuple{Tensor{Tuple{GradedString, GradedString}}, Tensor{Tuple{GradedString, GradedString}}}}, Int64} with 1 term:
-(gr"z"⊗gr"x")⊗(gr"w"⊗gr"y")
```
"""
macro regroup_str(s)
    ex = Meta.parse(s)
    Meta.isexpr(ex, :(->), 2) || error("invalid format")
    regroup(ex.args[1], ex.args[2].args[2])
end

"""
    regroup(a, b) -> Regroup

Return a `Regroup` object that can be used to rearrange the components of tensors and
possibly other structures.

!!! warning
    `regroup` is deprecated. Use the `regroup""` string macro instead.

See [`@regroup_str`](@ref).
"""
function regroup(a, b)
    d = Dict{Any,Int}()
    A = regroup_build!((d, Ref(0)), a)
    A isa Tuple || error("source must be a tuple")
    B = regroup_replace!(d, b)
    isempty(d) || error("incompatible target tree")
    Regroup{A,B}()
end

deg(::Regroup) = Zero()

"""
    regroup_inv(a, b) -> Tuple{Regroup, Regroup}

Return the tuple `(regroup(a, b), regroup(b, a))`.

!!! warning
    `regroup_inv` is deprecated. Use the `regroup_inv""` string macro instead.

See [`@regroup_inv_str`](@ref).
"""
regroup_inv(a, b) = (regroup(a, b), regroup(b, a))

"""
    regroup_inv"a -> b" -> Tuple{Regroup, Regroup}

Create the tuple `(regroup"a -> b", regroup"b -> a")` containing the `Regroup`
objects for transformations in both directions.

See also [`@regroup_str`](@ref).
"""
macro regroup_inv_str(s)
    ex = Meta.parse(s)
    Meta.isexpr(ex, :(->), 2) || error("invalid format")
    regroup_inv(ex.args[1], ex.args[2].args[2])
end

"""
    swap(t::AbstractTensor{Tuple{T1,T2}}) where {T1,T2}

This linear function swaps the components of two-component tensors. If the two components
of a tensor `t` have non-zero degrees, then the usual sign `(-1)^(deg(t[1])*deg(t[2]))` is introduced.
In this case the returned value is of type `Linear1` instead of `Tensor`.
By default, all terms have zero degree.

Note that `swap` is a special case of `regroup`:  it is simply defined as `regroup(:((1, 2)), :((2, 1)))`.

See also [`Tensor`](@ref), [`deg`](@ref), [`regroup`](@ref), [`$(@__MODULE__).DefaultCoefftype`](@ref).

# Examples

## Examples without degrees

```jldoctest
julia> t = Tensor("x", "z")
"x"⊗"z"

julia> swap(t)
"z"⊗"x"

julia> a = Linear("x" => 1, "yy" => 1) ⊗ Linear("z" => 1, "ww" => 1)
Linear{Tensor{Tuple{String, String}}, Int64} with 4 terms:
"x"⊗"z"+"x"⊗"ww"+"yy"⊗"z"+"yy"⊗"ww"

julia> swap(a)
Linear{Tensor{Tuple{String, String}}, Int64} with 4 terms:
"ww"⊗"yy"+"ww"⊗"x"+"z"⊗"x"+"z"⊗"yy"

julia> swap(a; coeff = 2)
Linear{Tensor{Tuple{String, String}}, Int64} with 4 terms:
2*"ww"⊗"yy"+2*"ww"⊗"x"+2*"z"⊗"x"+2*"z"⊗"yy"
```
## Examples with degrees

The degree of a `GradedString` (created with the `gr""` string macro) is its length.
```jldoctest
julia> using $(@__MODULE__).TestHelpers: GradedString, @gr_str

julia> t = Tensor(gr"x", gr"z")
gr"x"⊗gr"z"

julia> swap(t)
Linear1{Tensor{Tuple{GradedString, GradedString}}, Int64} with 1 term:
-gr"z"⊗gr"x"

julia> a = Linear(gr"x" => 1, gr"yy" => 1) ⊗ Linear(gr"z" => 1, gr"ww" => 1)
Linear{Tensor{Tuple{GradedString, GradedString}}, Int64} with 4 terms:
gr"x"⊗gr"z"+gr"yy"⊗gr"ww"+gr"x"⊗gr"ww"+gr"yy"⊗gr"z"

julia> swap(a)
Linear{Tensor{Tuple{GradedString, GradedString}}, Int64} with 4 terms:
-gr"z"⊗gr"x"+gr"ww"⊗gr"x"+gr"z"⊗gr"yy"+gr"ww"⊗gr"yy"
```
"""
const swap = regroup(:((1,2)), :((2,1)))

regroup_indices!(iv::Vector{Vector{Int}}, ii::Vector{Int}, ::Int) = push!(iv, copy(ii))

function regroup_indices!(iv::Vector{Vector{Int}}, ii::Vector{Int}, @nospecialize(t::Tuple))
    for (i, x) in enumerate(t)
        regroup_indices!(iv, push!(ii, i), x)
        pop!(ii)
    end
    iv
end

function regroup_getindex_expr(ii::Vector{Int})
    foldl(ii; init = :t) do ex, i
        Expr(:call, :regroup_getindex, ex, i)
    end
end

regroup_expr(iv::Vector{Vector{Int}}, i::Int) = regroup_getindex_expr(iv[i])

function regroup_expr(iv::Vector{Vector{Int}}, @nospecialize(t::Tuple))
    Expr(:call, :regroup_collect, (regroup_expr(iv, x) for x in t)...)
end

@generated function regroup_callable(rg::Regroup{A,B}, regroup_collect, t) where {A,B}
    iv = regroup_indices!(Vector{Int}[], Int[], A)
    regroup_expr(iv, B)
end

regroup_check_arg(::Type, ::Int, ::Type) = true

function regroup_check_arg(T::Type, A::Tuple, TX::Type)
    @foldable
    TX <: T && begin
        TT = fieldtypes(TX)
        length(TT) == length(A) && all(map(Fix1(regroup_check_arg, T), A, TT))
    end
end

@propagate_inbounds regroup_getindex(x, i) = x[i]
@propagate_inbounds regroup_getindex(::Type{T}, i) where T = fieldtypes(T)[i]

regroup_flatten(x) = (x,)
regroup_flatten(t::Tuple) = tuple_cat(map(regroup_flatten, t)...)
regroup_flatten(::Type{T}) where T <: AbstractTensor = regroup_flatten(fieldtypes(T))

#
# regrouping of tensors
#

@generated function regroup_tensor_signexp(rg::Regroup{A,B}, t) where {A,B}
    perm = regroup_flatten(B)
    N = length(perm)
    iv = regroup_indices!(Vector{Int}[], Int[], A)
    dv = [Expr(:call, deg, regroup_getindex_expr(iv[i])) for i in 1:N]
    dp = (Expr(:call, *, dv[i], dv[j]) for j in 1:N for i in 1:j-1 if perm[i] > perm[j])
    quote
        sum0(($(dp...),))
    end
end

regroup_tensor_nosign(rg, t) = regroup_callable(rg, Tensor∘tuple, t)

@linear_kw function regroup_tensor_sign(rg, t::T;
        coefftype = Sign,
        addto = zero(linear_return_type(rg, unval(coefftype), T)),
        coeff = ONE,
        is_filtered::Bool = false) where T <: AbstractTensor
    rgt = regroup_tensor_nosign(rg, t)
    c = has_char2(addto) ? coeff : withsign(regroup_tensor_signexp(rg, t), coeff)
    addmul!(addto, rgt, c; is_filtered)
end

function (rg::Regroup{A,B})(t::T; kw...) where {A, B, T <: AbstractTensor}
    regroup_check_arg(AbstractTensor, A, T) ||
        error("argument type $T does not match first Regroup parameter $A")
    if regroup_tensor_signexp_type(rg, T) === Zero
        regroup_tensor_nosign(rg, t; kw...)
    else
        regroup_tensor_sign(rg, t; kw...)
    end
end

@linear ::Regroup

keeps_filtered(::Regroup, ::Type{<:AbstractTensor}) = true

hastrait(rg::Regroup, trait::Val, ::Type{T}) where T <: AbstractTensor =
    regroup_tensor_signexp_type(rg, T) !== Zero && hastrait(regroup_tensor_sign, trait, typeof(rg), T)

function regroup_tensor_signexp_type(rg::Regroup, T::Type{<:AbstractTensor})
    @foldable
    perm = regroup_flatten(regroup_b(rg))
    DT = map(Fix1(return_type, deg), regroup_flatten(T))
    promote_type(Zero, (promote_type_product(DT[i], DT[j]) for j in 1:length(perm) for i in 1:j-1 if perm[i] > perm[j])...)
end

function return_type(rg::Regroup, T::Type{<:AbstractTensor})
    @foldable
    regroup_check_arg(AbstractTensor, regroup_a(rg), T) || return Union{}
    SE = regroup_tensor_signexp_type(rg, T)
    U = regroup_callable(rg, (TS...) -> Tensor{Tuple{TS...}}, T)
    SE === Zero && return U
    S = signtype(SE)
    Linear1{U, S <: Sign ? DefaultCoefftype : S}
end
