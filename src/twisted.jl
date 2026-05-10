
# twisted tensors

abstract type AbstractTwistedTensor{X,Y,TWC} <: AbstractTensor{Tuple{X,Y}} end

Base.Tuple(t::AbstractTwistedTensor) = (t.x, t.y)

export LeftTwistedTensor, RightTwistedTensor, lefttwistedtensor, righttwistedtensor

"""
    LeftTwistedTensor{X,Y,TWC}

See also [`RightTwistedTensor`](@ref), [`lefttwistedtensor`](@ref).
"""
struct LeftTwistedTensor{X,Y,TWC} <: AbstractTwistedTensor{X,Y,TWC}
    twc::TWC
    x::X
    y::Y
end

@struct_equal_hash LeftTwistedTensor

tensor_operator(::Type{<:LeftTwistedTensor}) = "⊗˱"

"""
    lefttwistedtensor(twc)

Return a callable object that converts an `AbstractTensor` to a `LeftTwistedTensor`
with twisting cochain `twc`. The callable object is linear.

See also [`LeftTwistedTensor`](@ref), [`righttwistedtensor`](@ref).

# Example
```jldoctest
julia> @linear twc;  # some twisting cochain

julia> t = Tensor("x", "y") |> lefttwistedtensor(twc)
"x"⊗˲"y"

julia> typeof(t)
LeftTwistedTensor{typeof(twc), String, String}
```
"""
lefttwistedtensor(twc) = TensorSplat(Fix1(LeftTwistedTensor, twc))

twistedtensor(t::LeftTwistedTensor) = lefttwistedtensor(t.twc)

function tensor_twisting(t::LeftTwistedTensor)
    convert(Tensor, t) |> Tensor(identity, coprod) |> regroup"(1, (2, 3)) -> ((1, 2), 3)" |>
        Tensor(Tensor(identity, t.twc), identity) |> Tensor(TensorSplat(*), identity) |> lefttwistedtensor(t.twc)
end

"""
    RightTwistedTensor{X,Y,TWC}

See also [`LeftTwistedTensor`](@ref),  [`righttwistedtensor`](@ref).
"""
struct RightTwistedTensor{X,Y,TWC} <: AbstractTwistedTensor{X,Y,TWC}
    twc::TWC
    x::X
    y::Y
end

@struct_equal_hash RightTwistedTensor

tensor_operator(::Type{<:RightTwistedTensor}) = "⊗˲"

"""
    righttwistedtensor(twc)

Return a callable object that converts an `AbstractTensor` to a `RightTwistedTensor`
with twisting cochain `twc`. The callable object is linear.

See also [`RightTwistedTensor`](@ref), [`lefttwistedtensor`](@ref).

# Example
```jldoctest
julia> @linear twc;  # some twisting cochain

julia> t = Tensor("x", "y") |> righttwistedtensor(twc)
"x"⊗˲"y"

julia> typeof(t)
RightTwistedTensor{typeof(twc), String, String}
```
"""
righttwistedtensor(twc) = TensorSplat(Fix1(RightTwistedTensor, twc))

function twistedtensor_diff_type(::Type{S}, t::LeftTwistedTensor{X,Y,TWC}) where {S,X,Y,TWC}
    L0 = return_type(diff, Tensor{Tuple{X,Y}})
    T0, R0 = _termtype(L0), _coefftype(L0)
    L1 = return_type(coprod, Y)
    T1, R1 = _termtype(L1), _coefftype(L1)
    L2 = return_type(t.twc, fieldtypes(T1)[2])
    T2, R2 = _termtype(L2), _coefftype(L2)
    L3 = return_type(*, X, T2)
    T3, R3 = _termtype(L3), _coefftype(L3)
    U0 = LeftTwistedTensor{fieldtypes(T0)..., TWC}
    U1 = LeftTwistedTensor{T3, fieldtypes(T1)[2], TWC}
    U = promote_typejoin(U0, U1)
    R = promote_type(S, R0, R1, R2, R3)
    Linear{U,R}
end

function twistedtensor_diff_type(::Type{S}, t::RightTwistedTensor{X,Y,TWC}) where {S,X,Y,TWC}
    L0 = return_type(diff, Tensor{Tuple{X,Y}})
    T0, R0 = _termtype(L0), _coefftype(L0)
    L1 = return_type(coprod, X)
    T1, R1 = _termtype(L1), _coefftype(L1)
    L2 = return_type(t.twc, fieldtypes(T1)[2])
    T2, R2 = _termtype(L2), _coefftype(L2)
    L3 = return_type(*, T2, Y)
    T3, R3 = _termtype(L3), _coefftype(L3)
    U0 = RightTwistedTensor{fieldtypes(T0)..., TWC}
    U1 = RightTwistedTensor{fieldtypes(T1)[1], T3, TWC}
    U = promote_typejoin(U0, U1)
    R = promote_type(S, R0, R1, R2, R3)
    Linear{U,R}
end

twistedtensor(t::RightTwistedTensor) = righttwistedtensor(t.twc)

function tensor_twisting(t::RightTwistedTensor)
    convert(Tensor, t) |> Tensor(coprod, identity) |> regroup"((1, 2), 3) -> (1, (2, 3))" |>
        Tensor(identity, Tensor(t.twc, identity)) |> Tensor(identity, TensorSplat(*)) |> righttwistedtensor(t.twc)
end

@linear_kw function diff(t::AbstractTwistedTensor;
        coefftype = DefaultCoefftype,
        addto = zero(twistedtensor_diff_type(unval(coefftype), t)),
        coeff = ONE)
    addmul!(addto, convert(Tensor, t) |> diff |> twistedtensor(t), coeff)
    addmul!(addto, t |> tensor_twisting, t isa LeftTwistedTensor ? coeff : -coeff)
    return addto
end
