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
Base.Tuple(::AbstractTensor)
LinearCombinations.cat
flatten
swap
Regroup
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
*(::AbstractTensor,::AbstractTensor)
coprod(::AbstractTensor)
LinearCombinations.diff(::AbstractTensor)
```
