from firedrake import *
from glueplex import make_periodic_mesh

n = 8
mesh = UnitCubeMesh(n, n, n, reorder=False)
File("mesh.pvd").write(mesh.coordinates)
mesh = UnitCubeMesh(n, n, n, reorder=False)

class Mapping():

    def is_slave(self, x):
        eps = 1e-8
        return (x[0] > 1-eps or x[1] > 1-eps or x[2] > 1-eps)

    def map_to_master(self, x):
        master = x.copy()
        eps = 1e-8
        if x[0] > 1-eps:
            master[0] -= 1
        if x[1] > 1-eps:
            master[1] -= 1
        if x[2] > 1-eps:
            master[2] -= 1
        return master

mapping = Mapping()
periodic_mesh = make_periodic_mesh(mesh, mapping)

W = VectorFunctionSpace(periodic_mesh, "CG", 1)
w = Function(W).project(SpatialCoordinate(periodic_mesh))
File("test.pvd").write(w)

# import IPython; IPython.embed()
