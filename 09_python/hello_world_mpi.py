#import the MPI class from mpi4py
from mpi4py import MPI
# call the COMM_WORLD attribute, store that in comm
comm = MPI.COMM_WORLD
# one of the attributes comm has is rank
print("Hello world, I am process: " + str(comm.rank))
