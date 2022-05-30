from mpi4py import MPI
import numpy
comm = MPI.COMM_WORLD
sendbuf = []

if comm.rank == 0:
    m = numpy.array(range(comm.size * comm.size), dtype=float)
    m.shape=(comm.size, comm.size)
    print("Original array on root process\n" + str(m))
    sendbuf = m
    
# first scatter like before
v = comm.scatter(sendbuf, root=0)
print("I got this data:\n" + str(v) + "\n and my ranks is " + str(comm.rank))
#do some work on each process and then gather back onto root
v = v * v
recvbuf = comm.gather(v, root=0)
if comm.rank == 0:
        print("New array on rank 0:\n" + str(numpy.array(recvbuf)))
