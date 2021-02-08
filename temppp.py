import math

n = 2**22
m = 257
d1 = sum(math.log2(x) for x in range(1,n+1))
d2 = sum(math.log2(x) for x in range(1, m))
u = sum(math.log2(x) for x in range(1, m+n))

print ((u-d1-d2)/8)