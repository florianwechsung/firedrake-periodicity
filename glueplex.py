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
    # print("dm_cells " , dm_cells)
    num_dm_vertices = dm_vertices[1] - dm_vertices[0]
    # print("dm_vertices " , dm_vertices)
    num_dm_edges = dm_edges[1] - dm_edges[0]
    # print("dm_edges " , dm_edges)
    if space_dim == 3:
        dm_facets = dm.getHeightStratum(1)
        num_dm_facets = dm_facets[1] - dm_facets[0]
        # print("dm_facets ", dm_facets)
    else:
        num_dm_facets = 0

    num_pdm_cells = num_dm_cells

    dm_coordinates = dm.getCoordinatesLocal()
    dm_section = dm.getCoordinateSection()

    def dm_coords(p):
        return dm.getVecClosure(
            dm_section, dm_coordinates, p).reshape(-1, space_dim).mean(axis=0)

    is_master = [None] * dm.getChart()[1]
    is_slave = [None] * dm.getChart()[1]
    coords = [None] * dm.getChart()[1]
    old_to_new_number = -np.ones(dm.getChart()[1], dtype=np.int)
    dm_slave_to_master = -np.ones(dm.getChart()[1], dtype=np.int)
    counter = 0
    for p in range(*dm_cells):
        old_to_new_number[p] = counter
        counter += 1

    def build_mapping(begin, end, counter):
        for p in range(begin, end):
            x = dm_coords(p)
            coords[p] = x
            if mapping.is_slave(x):
                is_slave[p] = True
            else:
                if mapping.is_master(x):
                    is_master[p] = True
                old_to_new_number[p] = counter
                counter += 1

        for p in range(begin, end):
            if old_to_new_number[p] == -1:
                x = coords[p]
                mapx = mapping.map_to_master(x)
                # print("Search for master for vertex x=", x)
                for q in range(begin, end):  # quadratic complexity, fix me
                    if p == q or not is_master[q]:
                        continue
                    y = coords[q]
                    if abs(mapx[0]-y[0]) < eps and abs(mapx[1]-y[1]) < eps and abs(mapx[2]-y[2]) < eps:
                        old_to_new_number[p] = old_to_new_number[q]
                        dm_slave_to_master[p] = q
                        break
                    if q == dm_vertices[1]-1:
                        raise RuntimeError(
                            "Could not find a master for slave vertex at " + str(x)
                            + "\nExpected master vertex at " + str(mapx))
        return counter
    import time
    start = time.time()
    counter = build_mapping(*dm_vertices, counter)
    num_pdm_vertices = counter - num_pdm_cells
    if space_dim == 3:
        counter = build_mapping(*dm_facets, counter)
        num_pdm_facets = counter - num_pdm_vertices - num_pdm_cells
    else:
        num_pdm_facets = 0
    counter = build_mapping(*dm_edges, counter)
    num_pdm_edges = counter - num_pdm_vertices - num_pdm_cells - num_pdm_facets
    end = time.time()
    print("Time for mapping", end-start)
    # print(np.vstack((np.asarray(list(range(*dm.getChart()))), old_to_new_number)).T)

    pdm_cells = (0, num_pdm_cells)
    pdm_vertices = (num_pdm_cells, num_pdm_cells + num_pdm_vertices)
    pdm_facets = (num_pdm_cells + num_pdm_vertices, num_pdm_cells + num_pdm_vertices + num_pdm_facets)
    pdm_edges = (num_pdm_cells + num_pdm_vertices + num_pdm_facets, num_pdm_cells + num_pdm_vertices + num_pdm_facets + num_pdm_edges)

    print("Number of cells", num_dm_cells, "->", num_pdm_cells)
    print("pdm_cells", (pdm_cells))
    print("Number of vertices", num_dm_vertices, "->", num_pdm_vertices)
    print("pdm_vertices", (pdm_vertices))
    print("Number of facets", num_dm_facets, "->", num_pdm_facets)
    print("pdm_facets", (pdm_facets))
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
    if space_dim == 3:
        for p in range(*pdm_facets):
            pdm.setConeSize(p, 3)
    pdm.setUp()

    def copy_label(label, rpoint, cpoint, new_name=None):
        if dm.hasLabel(label):
            if new_name is None:
                new_name = label
            pdm.setLabelValue(new_name, rpoint, dm.getLabelValue(label, cpoint))

    for dm_edge in range(*dm_edges):
        cone = dm.getCone(dm_edge)
        orient = dm.getConeOrientation(dm_edge)
        pdm_cone = cone.copy()
        for i in range(len(cone)):
            pdm_cone[i] = old_to_new_number[cone[i]]
        pdm.setCone(old_to_new_number[dm_edge], pdm_cone, orientation=orient)
        if dim == 2:
            if not is_slave[dm_edge]:
                copy_label("Face Sets", old_to_new_number[dm_edge], dm_edge)
            if is_master[dm_edge]:
                copy_label("exterior_facets", old_to_new_number[dm_edge], dm_edge, new_name="interior_facets")
            elif not is_slave[dm_edge]:
                copy_label("exterior_facets", old_to_new_number[dm_edge], dm_edge)

    if space_dim == 3:
        for dm_facet in range(*dm_facets):
            cone = dm.getCone(dm_facet)
            orient = dm.getConeOrientation(dm_facet)
            pdm_cone = cone.copy()
            for i in range(len(cone)):
                pdm_cone[i] = old_to_new_number[cone[i]]
            pdm.setCone(old_to_new_number[dm_facet], pdm_cone, orientation=orient)
            if not is_slave[dm_facet]:
                copy_label("Face Sets", old_to_new_number[dm_facet], dm_facet)
            if is_master[dm_facet]:
                copy_label("exterior_facets", old_to_new_number[dm_facet], dm_facet, new_name="interior_facets")
            elif not is_slave[dm_facet]:
                copy_label("exterior_facets", old_to_new_number[dm_facet], dm_facet)

    for dm_cell in range(*dm_cells):
        cone = dm.getCone(dm_cell)
        orient = dm.getConeOrientation(dm_cell)
        if all(not is_slave[c] for c in cone):
            pdm.setCone(old_to_new_number[dm_cell], [old_to_new_number[c] for c in cone], orientation=orient)
            continue
        if space_dim == 2:
            pdm_cone = [None, None, None]
            pdm_orient = [None, None, None]
            for i in range(len(cone)):
                pdm_cone[i] = old_to_new_number[cone[i]]
                if not is_slave[cone[i]]:
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
        else:
            # get the order of the vertices on a facet
            def get_vertex_walk_for_facet(facet, start):
                edges = pdm.getCone(facet)
                edges_orientation = pdm.getConeOrientation(facet)
                if start >= 0:
                    edge_cone = pdm.getCone(edges[start])
                    if edges_orientation[start] == 0:
                        vertex_walk = [edge_cone[0], edge_cone[1], None]
                    elif edges_orientation[start] == -2:
                        vertex_walk = [edge_cone[1], edge_cone[0], None]
                    else:
                        raise NotImplementedError
                    other_edge = (start + 1)%3
                    other_vertices = pdm.getCone(edges[other_edge])
                    if other_vertices[0] in vertex_walk:
                        vertex_walk[2] = other_vertices[1]
                    else:
                        vertex_walk[2] = other_vertices[0]
                else:
                    start_edge = -(start+1)
                    edge_cone = pdm.getCone(edges[start_edge])
                    if edges_orientation[start_edge] == 0:
                        vertex_walk = [edge_cone[1], edge_cone[0], None]
                    elif edges_orientation[start_edge] == -2:
                        vertex_walk = [edge_cone[0], edge_cone[1], None]
                    else:
                        raise NotImplementedError
                    other_edge = (start_edge + 1)%3
                    other_vertices = pdm.getCone(edges[other_edge])
                    if other_vertices[0] in vertex_walk:
                        vertex_walk[2] = other_vertices[1]
                    else:
                        vertex_walk[2] = other_vertices[0]
                return vertex_walk

            vertex_walk = get_vertex_walk_for_facet(old_to_new_number[cone[0]], 0)
            v0 = vertex_walk[0]
            v1 = vertex_walk[1]
            v2 = vertex_walk[2]
            v3 = old_to_new_number[[i for i in dm.getTransitiveClosure(dm_cell, useCone=True)[0][-4:] if old_to_new_number[i] not in vertex_walk][0]]
            pdm_cone = [old_to_new_number[cone[0]], None, None, None]
            pdm_orient = [0, None, None, None]
            for i in range(1, 4):
                new_facet = old_to_new_number[cone[i]]
                vertex_walk = get_vertex_walk_for_facet(new_facet, 0)
                if v2 not in vertex_walk:
                    pdm_cone[1] = new_facet
                    for start in range(-3, 3):
                        walk = get_vertex_walk_for_facet(new_facet, start)
                        if walk[0] == v0 and walk[1] == v3 and walk[2] == v1:
                            pdm_orient[1] = start
                            break
                elif v1 not in vertex_walk:
                    pdm_cone[2] = new_facet
                    for start in range(-3, 3):
                        walk = get_vertex_walk_for_facet(new_facet, start)
                        if walk[0] == v0 and walk[1] == v2 and walk[2] == v3:
                            pdm_orient[2] = start
                            break
                elif v0 not in vertex_walk:
                    pdm_cone[3] = new_facet
                    for start in range(-3, 3):
                        walk = get_vertex_walk_for_facet(new_facet, start)
                        if walk[0] == v2 and walk[1] == v1 and walk[2] == v3:
                            pdm_orient[3] = start
                            break
                else:
                    raise NotImplementedError
            
            if False:
                should_be = [[0, 1, 2], [0, 3, 1], [0, 2, 3], [2, 1, 3]]
                mapdict = {v0: 0, v1: 1, v2: 2, v3: 3}
                print("---")
                for i in range(4):
                    walk = get_vertex_walk_for_facet(pdm_cone[i], pdm_orient[i])
                    # print("Fast", walk)
                    #walk = get_vertex_walk_for_facet_slow(new_cell_cone[i], new_cell_cone_orientation[i])
                    #print("Slow", walk)
                    renamed_walk = [mapdict[w] for w in walk]
                    print("renamed_walk", renamed_walk)
                    for j in range(3):
                        assert renamed_walk[j] == should_be[i][j]
                    #print("Vertex walk for cell %i, edge %i: %s. " % (new_cell, new_cell_cone[i], renamed_walk))
        # print(f"setCone({old_to_new_number[dm_cell]}, {pdm_cone}, orientation={pdm_orient})")
        pdm.setCone(old_to_new_number[dm_cell], pdm_cone, orientation=pdm_orient)

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
        if not is_slave[old_vertex]:
            idx = old_to_new_number[old_vertex] - pdm_vertices[0]
            pdm_coords_array[idx] = old_coords

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
    dim = new_mesh.topological_dimension()
    for cell in range(len(V.cell_node_list)):
        old_vertices_in_cell = mesh.cell_closure[cell][0:(dim+1)]
        new_vertices_in_cell = new_mesh.cell_closure[cell][0:(dim+1)]
        old_vertices_in_new_numbering = list(
            [old_to_new_number[p] for p in old_vertices_in_cell]
        )
        if not (set(new_vertices_in_cell) == set(old_vertices_in_new_numbering)):
            raise RuntimeError(
                "Vertices in cell %i not the same after applying periodicity."
                "\nDid you forget to pass `reorder=False` to the Mesh"
                " constructor?" % cell
            )

        for i in range(dim+1):
            j = old_vertices_in_new_numbering.index(new_vertices_in_cell[i])
            b[Vp.cell_node_list[cell][i], :] = a[V.cell_node_list[cell][j], :]
    return fd.Mesh(coordsp, reorder=False)
