import numpy as np
from firedrake.petsc import PETSc
import firedrake as fd
eps = 1e-6


def generate_periodic_plex(dm, mapping):
    space_dim = dm.getCoordinateDim()
    dm_cells = dm.getHeightStratum(0)
    dm_vertices = dm.getDepthStratum(0)
    dm_edges = dm.getDepthStratum(1)

    num_dm_cells = dm_cells[1] - dm_cells[0]
    num_dm_vertices = dm_vertices[1] - dm_vertices[0]
    num_dm_edges = dm_edges[1] - dm_edges[0]

    num_pdm_cells = num_dm_cells

    dm_coordinates = dm.getCoordinatesLocal()
    dm_section = dm.getCoordinateSection()

    def dm_coords(p):
        return dm.getVecClosure(
            dm_section, dm_coordinates, p).reshape(-1, space_dim).mean(axis=0)

    is_master = [None] * dm.getChart()[1]
    old_to_new_number = -np.ones(dm.getChart()[1], dtype=np.int)
    dm_slave_to_master = -np.ones(dm.getChart()[1], dtype=np.int)
    counter = 0
    for p in range(*dm_cells):
        old_to_new_number[p] = counter
        counter += 1

    for p in range(*dm_vertices):
        x = dm_coords(p)
        if not mapping.is_slave(x):
            old_to_new_number[p] = counter
            counter += 1
            is_master[p] = True
        else:
            is_master[p] = False

    for p in range(*dm_vertices):
        if old_to_new_number[p] == -1:
            x = dm_coords(p)
            mapx = mapping.map_to_master(x)
            print("Search for master for vertex x=", x)
            for q in range(*dm_vertices):  # quadratic complexity, fix me
                if p == q:
                    continue
                y = dm_coords(q)
                if np.linalg.norm(mapx-y) < eps:
                    old_to_new_number[p] = old_to_new_number[q]
                    dm_slave_to_master[p] = q
                    print("Master found at y=", y)
                    break
                if q == dm_vertices[1]-1:
                    raise RuntimeError

    num_pdm_vertices = counter - num_pdm_cells

    for p in range(*dm_edges):
        x = dm_coords(p)
        if not mapping.is_slave(x):
            old_to_new_number[p] = counter
            counter += 1
            is_master[p] = True
        else:
            is_master[p] = False
    for p in range(*dm_edges):
        x = dm_coords(p)
        if old_to_new_number[p] == -1:  # quadratic complexity, fix me
            mapx = mapping.map_to_master(x)
            for q in range(*dm_edges):
                if p == q:
                    continue
                y = dm_coords(q)
                if np.linalg.norm(mapx-y) < eps:
                    old_to_new_number[p] = old_to_new_number[q]
                    dm_slave_to_master[p] = q
                    print(f"Found master for edge {p}({x}): {q}({y})")
                    break
                if q == dm_edges[1]-1:
                    raise RuntimeError

    print(np.vstack((np.asarray(list(range(*dm.getChart()))), old_to_new_number)).T)

    num_pdm_edges = counter - num_pdm_vertices - num_pdm_cells
    pdm_cells = (0, num_pdm_cells)
    pdm_vertices = (num_pdm_cells, num_pdm_cells + num_pdm_vertices)
    pdm_edges = (num_pdm_cells+num_pdm_vertices, num_pdm_cells+num_pdm_vertices+num_pdm_edges)

    print("Number of cells", num_dm_cells, "->", num_pdm_cells)
    print("pdm_cells", (pdm_cells))
    print("Number of vertices", num_dm_vertices, "->", num_pdm_vertices)
    print("pdm_vertices", (pdm_vertices))
    print("Number of edges", num_dm_edges, "->", num_pdm_edges)
    print("pdm_edges", (pdm_edges))
    pdm = PETSc.DMPlex().create(comm=dm.comm)
    dim = dm.getDimension()
    pdm.setChart(0, pdm_edges[1])
    pdm.setDimension(dim)
    for p in range(*pdm_cells):
        pdm.setConeSize(p, dim+1)
    for p in range(*pdm_edges):
        pdm.setConeSize(p, 2)
    pdm.setUp()

    for dm_edge in range(*dm_edges):
        cone = dm.getCone(dm_edge)
        orient = dm.getConeOrientation(dm_edge)
        pdm_cone = cone.copy()
        for i in range(len(cone)):
            pdm_cone[i] = old_to_new_number[cone[i]]
        pdm.setCone(old_to_new_number[dm_edge], pdm_cone, orientation=orient)
        print(f"setCone({old_to_new_number[dm_edge]}, {pdm_cone}, orientation={orient})")

    for dm_cell in range(*dm_cells):
        cone = dm.getCone(dm_cell)
        orient = dm.getConeOrientation(dm_cell)
        pdm_cone = cone.copy()
        pdm_orient = orient.copy()
        for i in range(len(cone)):
            pdm_cone[i] = old_to_new_number[cone[i]]
            if is_master[cone[i]]:
                pdm_orient[i] = orient[i]
            else:
                dm_vertices_on_slave_edge = [p for p in dm.getTransitiveClosure(cone[i])[0] if dm_vertices[0] <= p < dm_vertices[1]]
                dm_vertices_on_master_edge = [p for p in dm.getTransitiveClosure(dm_slave_to_master[cone[i]])[0] if dm_vertices[0] <= p < dm_vertices[1]]
                assert len(dm_vertices_on_slave_edge) == 2
                assert len(dm_vertices_on_master_edge) == 2
                if old_to_new_number[dm_vertices_on_slave_edge[0]] == old_to_new_number[dm_vertices_on_master_edge[0]]:
                    assert old_to_new_number[dm_vertices_on_slave_edge[1]] == old_to_new_number[dm_vertices_on_master_edge[1]]
                    pdm_orient[i] = orient[i]
                elif old_to_new_number[dm_vertices_on_slave_edge[0]] == old_to_new_number[dm_vertices_on_master_edge[1]]:
                    assert old_to_new_number[dm_vertices_on_slave_edge[1]] == old_to_new_number[dm_vertices_on_master_edge[0]]
                    if orient[i] == 0:
                        pdm_orient[i] = -2
                    elif orient[i] == -2:
                        pdm_orient[i] = 0
                    else:
                        raise RuntimeError
                else:
                    raise RuntimeError
        pdm.setCone(old_to_new_number[dm_cell], pdm_cone, orientation=pdm_orient)
        print(f"setCone({old_to_new_number[dm_cell]}, {pdm_cone}, orientation={pdm_orient})")

    pdm.symmetrize()
    pdm.stratify()

    pdm.setCoordinateDim(space_dim)
    pdm_section = pdm.getCoordinateSection()

    pdm_section.setNumFields(1)
    pdm_section.setFieldComponents(0, space_dim)
    pdm_section.setChart(*pdm_vertices)
    for p in range(*pdm_vertices):
        pdm_section.setDof(p, space_dim)
        pdm_section.setFieldDof(p, 0, space_dim)
    pdm_section.setUp()

    pdm_coorddm = pdm.getCoordinateDM()
    pdm_coords = pdm_coorddm.createLocalVector()
    pdm_coords.setBlockSize(space_dim)
    pdm_coords.setName("coordinates")

    pdm_coords_array_ = pdm_coords.getArray()
    pdm_coords_array = pdm_coords_array_.reshape(-1, space_dim)

    # Now set the damn coordinates.
    for old_vertex in range(*dm_vertices):
        old_coords = dm_coords(old_vertex)
        if is_master[old_vertex]:
            idx = old_to_new_number[old_vertex] - pdm_vertices[0]
            pdm_coords_array[idx] = old_coords
        else:
            print("Ignore", old_coords)

    pdm_coords.setArray(pdm_coords_array_)
    pdm.setCoordinatesLocal(pdm_coords)

    return pdm, old_to_new_number


def make_periodic_mesh(mesh, mapping):
    plex = mesh._plex
    V = fd.VectorFunctionSpace(mesh, "DG", 1)
    coords = fd.Function(V).interpolate(fd.SpatialCoordinate(mesh))

    rebuilt_plex, old_to_new_number = generate_periodic_plex(plex, mapping)

    new_mesh = fd.Mesh(rebuilt_plex, reorder=False)
    Vp = fd.VectorFunctionSpace(new_mesh, "DG", 1)

    coordsp = fd.Function(Vp)
    a = coords.vector()
    b = coordsp.vector()
    for cell in range(len(V.cell_node_list)):
        old_vertices_in_cell = mesh.cell_closure[cell][0:3]
        new_vertices_in_cell = new_mesh.cell_closure[cell][0:3]
        old_vertices_in_new_numbering = list(
            [old_to_new_number[p] for p in old_vertices_in_cell]
        )
        assert(set(new_vertices_in_cell) == set(old_vertices_in_new_numbering))
        for i in range(3):
            j = old_vertices_in_new_numbering.index(new_vertices_in_cell[i])
            b[Vp.cell_node_list[cell][i], :] = a[V.cell_node_list[cell][j], :]
    return fd.Mesh(coordsp, reorder=False)
