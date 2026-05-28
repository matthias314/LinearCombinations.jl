```@meta
DocTestSetup = doctestsetup
```

# Tensors

```@docs
AbstractTensor
```

## Constructors

```@docs
Tensor
tensor
```

## Manipulating tensors

```@docs
Base.fieldtypes
Base.Tuple(::AbstractTensor)
LinearCombinations.cat
flatten
swap
Regroup
@regroup_str
@regroup_inv_str
regroup
regroup_inv
Base.transpose
```

## Calling tensors

```@docs
AbstractTensor(::AbstractTensor)
```

## Other functions accepting tensors

```@docs
deg(::AbstractTensor)
*(::AbstractTensor{NTuple{2,Any}},::AbstractTensor{NTuple{2,Any}})
coprod(::AbstractTensor)
LinearCombinations.diff(::AbstractTensor)
```

## [Twisted tensors](@id sec-twisted)

```@docs
LeftTwistedTensor
RightTwistedTensor
lefttwistedtensor
righttwistedtensor
```
