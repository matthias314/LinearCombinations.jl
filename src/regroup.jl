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

struct Regroup{A,B} end

# == is ===
# hash is computed from objectid

regroup_get(::Type{T}, field) where T <: Regroup = regroup_cache[T][field]

show(io::IO, rg::Regroup{A,B}) where {A,B} = print(io, "Regroup{$A,$B}")

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

regroup_inv(a, b) = (regroup(a, b), regroup(b, a))

const swap = regroup(:((1,2)), :((2,1)))

@propagate_inbounds @generated regroup_eval_expr(rg::Regroup, f, g, x) = regroup_get(rg, :expr)

#
# regrouping of tensors
#

_getindex(x) = x
@propagate_inbounds _getindex(x, i) = x[i]
@propagate_inbounds _getindex(x, i, ii...) = _getindex(_getindex(x, i), ii...)

@propagate_inbounds _getindex(::Type{Tensor{T}}, i) where T <: Tuple = T.parameters[i]

build_tensor_type(T...) = Tensor{Tuple{T...}}

function regroup_tensor_coeff_type(T...)
    RS = map(Fix1(sign_type ∘ return_type, deg), T)
    promote_type(Int, RS...)
end

function regroup_tensor_type(rg::Regroup, ::Type{T}, coefftype) where T <: Tensor
    U = regroup_eval_expr(rg, _getindex, build_tensor_type, T)
    Linear1{U,coefftype}
end

# @assume_effects allows to omit the sign computation for has_char2 coefficients
Base.@assume_effects :foldable :nothrow @generated function regroup_tensor_signexp(rg, f, x)
    args = Expr(:tuple, regroup_get(rg, :args)...)
    perm = regroup_get(rg, :perm)
    invdeg = (:(degs[$i]*degs[$j]) for j in 2:length(perm) for i in 1:j-1 if perm[i] > perm[j])
    quote
        degs = $args
        sum0(($(invdeg...),))
    end
end

@linear_kw function (rg::Regroup)(x::T;
        coefftype = regroup_tensor_coeff_type(map(typeof, factors(x))...),
        addto = zero(regroup_tensor_type(rg, T, coefftype)),
        coeff = ONE,
        is_filtered::Bool = false) where T <: Tensor
    addmul!(addto, regroup_eval_expr(rg, _getindex, Tensor, x),
            signed(regroup_tensor_signexp(rg, deg ∘ _getindex, x), coeff); is_filtered)
        # @inbounds has no effect for building the tensor or computing the sign
        # because we access tuples at (after code generation) fixed indices
end

return_type(rg::Regroup, ::Type{T}) where T <: Tensor =
    regroup_tensor_type(rg, T, regroup_tensor_coeff_type(T.parameters[1].parameters...))
