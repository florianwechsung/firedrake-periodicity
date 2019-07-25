from firedrake import *
from glueplex import make_periodic_mesh

# mesh = UnitSquareMesh(30, 30, reorder=False)
mesh = Mesh("square_with_hole.msh", reorder=False)

class Mapping():

    def is_slave(self, x):
        eps = 1e-8
        return (x[0] > 1-eps or x[1] > 1-eps)

    def map_to_master(self, x):
        master = x.copy()
        eps = 1e-8
        if x[0] > 1-eps:
            master[0] -= 1
        if x[1] > 1-eps:
            master[1] -= 1
        return master

mapping = Mapping()
periodic_mesh = make_periodic_mesh(mesh, mapping)

W = VectorFunctionSpace(periodic_mesh, "CG", 1)
w = Function(W).interpolate(SpatialCoordinate(periodic_mesh))
File("test.pvd").write(w)
