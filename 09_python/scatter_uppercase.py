<TO_DO>
import numpy as np

comm = <TO_DO>
rank = <TO_DO>
size = <TO_DO>

A = np.zeros((size,size))
if rank==0:
    A = np.random.randn(size, size)
    print("Original array on root process\n", A)
local_a = np.zeros(size)

comm.<TO_DO>(A, local_a, root=0)
print("Process", rank, "received", local_a)
