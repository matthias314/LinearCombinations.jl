```@meta
DocTestSetup = doctestsetup
```

# Linear and multilinear functions

## Linear extensions

```@docs
@linear
LinearExtension
matrixrepr
matrixrepr!
LinearCombinations.diff
coprod
```

## Multilinear extensions

```@docs
@multilinear
MultilinearExtension
LinearCombinations.mul
```

## Common functionality

```@docs
@linear_kw
LinearCombinations.DefaultCoefftype
linear_filter
keeps_filtered
LinearCombinations.termcoeff
LinearCombinations.has_coefftype
LinearCombinations.has_addto_coeff
LinearCombinations.has_isfiltered
LinearCombinations.has_sizehint
```

## Switching between linear and multilinear maps

```@docs
TensorSlurp
TensorSplat
```
