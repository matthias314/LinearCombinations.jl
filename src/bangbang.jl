module BangBang

export add!!, sub!!, inc!!, dec!!, addmul!!

using ..LinearCombinations: Linear, add!, sub!, addmul!

add!!(x, y) = x+y
add!!(x::Linear, y) = add!(x, y)

inc!!(x) = add!!(x, 1)

sub!!(x, y) = x-y
sub!!(x::Linear, y) = sub!(x, y)

dec!!(x) = add!!(x, -1)

addmul!!(x, y, c) = x+c*y
addmul!!(x::Linear, y, c) = addmul!(x, y, c)

end # module