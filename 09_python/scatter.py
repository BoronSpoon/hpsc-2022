from mpi4py import MPI
import numpy
comm = MPI.COMM_WORLD
sendbuf = []
if comm.rank == 0:
    m = numpy.random.randn(comm.size, comm.size)
    print("Original array on root process\n" + str(m))
    sendbuf = m
# call this on every rank, including rank 0
v = comm.scatter(sendbuf, root=0)
print("I got this data:\n" + str(v) + "\n and my ranks is " + str(comm.rank))
