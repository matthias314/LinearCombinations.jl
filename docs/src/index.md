```@meta
DocTestSetup = doctestsetup
```

# Overview

```@docs
LinearCombinations
```

Let us give a few simple examples illustrating the main features of the package.
For simplicity we use `Char` and `String` as term types and integers, rationals or
floating-point numbers as coefficients throughout this documentation.

We start with some linear combinations. They are of type [`Linear`](@ref), which can store
any number of term-coefficient pairs. Term type and coefficient type are automatically
determined to be `Char` and `Int` in the following example.
```@repl intro
using LinearCombinations
a = Linear('x' => 1, 'y' => 2)
typeof(a)
b = Linear('z' => 3, 'w' => -1)
c = a + 2*b - 'v'
c['y'], c['u']
```
Linear combinations with non-concrete types are also possible.
```@repl intro
p = Linear{AbstractVector,Real}([1,2,3] => 5, 4:6 => 1//2)
[4,5,6] - 2*p   # [4,5,6] is equal to 4:6
```
Next we define a linear function mapping terms to terms. This is done with the help of the macro [`@linear`](@ref).
```@repl intro
@linear f; f(x::Char) = uppercase(x)
f(a)
```
Another linear function, this time mapping terms to linear combinations.
```@repl intro
@linear g; g(x::Char) = Linear(f(x) => 1, x => -1)
g(a)
```
Multiplication is bilinear by default.
Recall that multiplying `Char` or `String` values in Julia means concatenation.
```@repl intro
a * 'w'
a * b
```
The next example is a user-defined bilinear function. Bilinearity is achieved by the macro [`@multilinear`](@ref).
```@repl intro
@multilinear h; h(x, y) = x*y*x
h(a, b)
```
Here is a user-defined multilinear function with a variable number of arguments.
```@repl intro
@multilinear j; j(x::Char...) = *(x...)
j(a)
j(a, b)
j(a, b, a)
```
In the following example we define a tensor and swap the two components of each summand.
```@repl intro
t = tensor(a, b)
swap(t)
```
We finally take the tensor product of the functions `f` and `g` and apply it to `t`.
```@repl intro
const k = Tensor(f, g)
k(Tensor('x', 'z'))
k(t)
```
