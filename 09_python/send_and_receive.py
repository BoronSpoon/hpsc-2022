from mpi4py import MPI
import numpy
comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size
a = numpy.array([rank]*10, dtype=float)
if rank == 0:
    comm.send(a, dest = (rank + 1) % size)
if rank > 0:
    data = comm.recv(source = (rank - 1) % size)
    comm.send(a, dest = (rank + 1) % size)
if rank == 0:
    data = comm.recv(source = size - 1)
print("My ranks is " + str(rank) + "\n and I received this array:\n" + str(data))
