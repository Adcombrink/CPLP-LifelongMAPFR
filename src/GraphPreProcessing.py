import math
import numpy as np
import networkx as nx
from LineSegmentIntersections import get_overlapping_edges


class GraphPreProcessor:

    def __init__(self, graph, r, s):
        self.graph = graph
        self.r = r  # agent radius
        self.s = s  # agent speed

        self.V_vvc = None
        self.V_vec = None
        self.E_vec = None
        self.E_eec = None
        self.dist = dict(nx.all_pairs_dijkstra_path_length(self.graph, weight='weight'))

    def edge_coords(self, edge):
        """ Returns the coordinates of the edge's vertices.

        Args:
            edge: edge label

        Returns:
            v1, v2: np.array, edge-end-point coordinates.
        """
        assert edge in self.graph.edges, f'Edge {edge} not in graph.'
        return np.array(self.graph.nodes[edge[0]]['pos']), np.array(self.graph.nodes[edge[1]]['pos'])

    def annotate_graph_with_ctc(self):
        """ Annotates the graph with non-zero unsafe intervals for each vertex-vertex, vertex-edge, and edge-edge pair.

        Based on the algorithm from Kasaura 2022, with some minor modifications:
            - V_vvc included, containing vertex-pairs within 2r distance.
            - V_vec contains the unsafe intervals for vertices given an edge that is traversed at t=0. This simplifies
              lookup downstream.

        This function populates the class variables
            V_vvc: dict, keys: vertices, values: set of vertices within 2r distance.
            V_vec: dict, keys: vertices, values: dict, keys: edges, values: unsafe interval.
            E_vec: dict, keys: vertices, values: dict, keys: vertices, values: unsafe interval.
            E_eec: dict, keys: edges, values: dict, keys: edges, values: unsafe interval.
        """
        vertex_hash_grid = self.get_vertex_hash_grid( 2 *self.r)

        P_vec = set()
        P_eec = set()
        C_vec = set()
        C_eec = set()

        # get the set of vertices within 2r distance (including self) of each vertex
        self.V_vvc = self.fixed_radius_near_neighbour( 2 *self.r, vertex_hash_grid)

        # get all vertex-edge pairs where the vertex is within 2r from the edge.
        #   1. vertex within 2r from the edge's end points
        for v in self.V_vvc:
            for u in self.V_vvc[v]:
                for e in set(self.graph.out_edges(u)) | set(self.graph.in_edges(u)):
                    P_vec.add((v, e))

        #   2. vertex within 2r from the edge (perpendicularly)
        for e in self.graph.edges:
            for v in self.perpendicular_rectangle(e, 2* self.r, vertex_hash_grid):
                P_vec.add((v, e))

        # get all edge-edge pairs within 2r from each other
        #   1. edges with an end point within 2r from the other
        for v, e1 in P_vec:
            for e2 in set(self.graph.out_edges(v)) | set(self.graph.in_edges(v)):
                P_eec.add((e1, e2))
                P_eec.add((e2, e1))
        #   2. edges that cross each other
        PC = get_overlapping_edges({e: self.edge_coords(e) for e in self.graph.edges}, 8)
        for e1 in PC:
            for e2 in PC[e1]:
                P_eec.add((e1, e2))

        # get vertex-edge unsafe intervals
        for v, e in P_vec:
            I = self.vertex_edge_unsafe_interval(v, e)
            if I is not None:
                C_vec.add((v, e, float(I[0]), float(I[1])))

        # get edge-edge unsafe intervals
        for e1, e2 in P_eec:
            I = self.edge_edge_unsafe_interval(e1, e2)
            if I is not None:
                C_eec.add((e1, e2, float(I[0]), float(I[1])))

        # Create filter functions
        #   1. Mapping: v -> e -> unsafe interval at e if v occupied at t=0
        #   2. Mapping: e -> v -> unsafe interval at v if e is traversed at t=0
        self.E_vec = {v: dict() for v in self.graph.nodes}
        self.V_vec = {e: dict() for e in self.graph.edges}
        for v, e, tao0, tao1 in C_vec:
            self.E_vec[v][e] = (tao0, tao1)
            self.V_vec[e][v] = (-tao1, -tao0)

        #   3. Mapping: e -> e -> unsafe interval at e2 if e1 is traversed at t=0
        self.E_eec = {e: dict() for e in self.graph.edges}
        for e1, e2, tao0, tao1 in C_eec:
            self.E_eec[e1][e2] = (tao0, tao1)

    def get_vertex_hash_grid(self, d):
        """ Assigns vertices to cells in a hash-grid, each with side length d.

        Source: D.M. Mount 2011 CMSC 754 Computational Geometry.

        Args:
            d: float, grid cell side length.
        Returns:
            hash_grid: dictionary of sets, keys: grid cell coordinates, values: set of vertices.
        """

        hash_grid = dict()

        for v in self.graph.nodes:
            x, y = self.graph.nodes[v]['pos']
            cell = (math.floor(x / d), math.floor(y / d))
            if cell not in hash_grid:
                hash_grid[cell] = set()
            hash_grid[cell].add(v)

        return hash_grid

    def fixed_radius_near_neighbour(self, d, vertex_hash_grid):
        """ Gets all vertex-vertex-pairs that are within a distance d from each other.

        This includes same vertex pairs.
        First, a hash-grid of size d is created and vertices are assigned to the cells.
        Then, each vertex is tested against all vertices in its current and neighbouring cells.

        Args:
            d: float, distance threshold
            vertex_hash_grid: dict, keys: cell coordinates, values: set of vertices in the cell.

        Returns:
            near_neighbours: dict, keys: vertices, values: set of vertices within d distance.
        """

        near_neighbours = {v: set() for v in self.graph.nodes}

        checked_vertices = set()
        for v in self.graph.nodes:

            # add self
            near_neighbours[v].add(v)

            # get the grid cell containing v and the neighbouring cells
            x, y = self.graph.nodes[v]['pos']
            curr_cell = (math.floor(x / d), math.floor(y / d))
            cells = [(curr_cell[0] + i, curr_cell[1] + j) for i in [-1, 0, 1] for j in [-1, 0, 1]]

            checked_vertices.add(v)  # by already adding v to the checked vertices, we avoid self-intersection

            for cell in cells:
                for u in vertex_hash_grid.get(cell, []):

                    if u in checked_vertices:
                        continue

                    if np.linalg.norm(np.array(self.graph.nodes[u]['pos']) - np.array(self.graph.nodes[v]['pos'])) <= d:
                        near_neighbours[v].add(u)
                        near_neighbours[u].add(v)

        return near_neighbours

    def perpendicular_rectangle(self, edge, d, vertex_hash_grid):
        """ Returns the vertices that are within the rectangle formed by the edge shifted d to either side.

        Assumption:
            The rectangle width d is <= the vertex_hash_grid cell side length.

        Args:
            edge: edge label
            d: float, width of the rectangle
            vertex_hash_grid: dict, keys: cell coordinates, values: set of vertices in the cell.

        Returns:
            vertices_within_rectangle: set of vertices within the rectangle.
        """

        def cross_product(v1, v2):
            """ Implemented since numpy cross is depreciated for 2D vectors."""
            return v1[0] * v2[1] - v1[1] * v2[0]

        def cells_intersecting_edge(e, d):
            """ Returns the r-grid cells that intersect with the edge e.

            Using only hash-grid coordinates. Start by adding the cell of the edge's start point. Then, move along the
            edge direction until a new cell is reached, add that cell, and repeat until the end point is reached.

            Args:
                e: tuple, edge-end-point coordinates.
                d: float, grid cell side length.
            Returns:
                intersected_cells: set of grid cell coordinates.
            """

            # get edge params in hash-grid coordinates
            e_start = e[0] / d
            e_end = e[1] / d
            direction = (e_end - e_start) / np.linalg.norm(e_end - e_start)

            # add the start cell (but get the end cell for checking termination)
            start_cell = (math.floor(e_start[0]), math.floor(e_start[1]))
            end_cell = (math.floor(e_end[0]), math.floor(e_end[1]))
            intersected_cells = {start_cell}

            # move along the line, adding each new cell as it is reached. Terminate when the end cell is reached.
            curr_v = e_start
            curr_cell = start_cell
            while curr_cell != end_cell:

                # get the vertical and horisontal boundary lines of the current cell in the direction of the edge
                v_boundary_x = curr_cell[0] + 1 if direction[0] >= 0 else curr_cell[0]
                v_boundary_line = ((v_boundary_x, curr_cell[1]), (0, 1))
                h_boundary_y = curr_cell[1] + 1 if direction[1] >= 0 else curr_cell[1]
                h_boundary_line = ((curr_cell[0], h_boundary_y), (1, 0))

                # get intersections with the boundary lines
                v_boundary_intersection = None
                if cross_product(direction, v_boundary_line[1]) != 0:
                    t = cross_product(v_boundary_line[0] - curr_v, v_boundary_line[1]) / cross_product(direction,
                                                                                                       v_boundary_line[
                                                                                                           1])
                    v_boundary_intersection = curr_v + t * direction
                h_boundary_intersection = None
                if cross_product(direction, h_boundary_line[1]) != 0:
                    t = cross_product(h_boundary_line[0] - curr_v, h_boundary_line[1]) / cross_product(direction,
                                                                                                       h_boundary_line[
                                                                                                           1])
                    h_boundary_intersection = curr_v + t * direction

                # determine which boundary line is intersected first, determining which cell is entered next
                if v_boundary_intersection is not None and round(v_boundary_line[0][1], 6) <= round(
                        v_boundary_intersection[1], 6) < round(v_boundary_line[0][1] + 1, 6):
                    curr_v = v_boundary_intersection
                    curr_cell = (curr_cell[0] + 1 if direction[0] >= 0 else curr_cell[0] - 1, curr_cell[1])
                else:
                    curr_v = h_boundary_intersection
                    curr_cell = (curr_cell[0], curr_cell[1] + 1 if direction[1] >= 0 else curr_cell[1] - 1)
                intersected_cells.add(curr_cell)

            return intersected_cells

        # get edge params
        e = self.edge_coords(edge)
        v1, v2 = e
        l = np.linalg.norm((v2 - v1))
        direction = (v2 - v1) / l

        # get supporting edges
        pdir = np.array([-direction[1], direction[0]])
        ebar_1 = (v1 + pdir * d, v2 + pdir * d)
        ebar_2 = (v1 - pdir * d, v2 - pdir * d)

        # get all cells that intersect with the rectangle
        intersecting_cells = cells_intersecting_edge(e, d)
        intersecting_cells.update(cells_intersecting_edge(ebar_1, d))
        intersecting_cells.update(cells_intersecting_edge(ebar_2, d))

        # get all vertices in the intersecting cells
        vertices_to_check = set()
        for cell in intersecting_cells:
            vertices_to_check.update(vertex_hash_grid.get(cell, set()))

        # get all vertices that intersect the rectangle
        vertices_within_rectangle = set()
        for u in vertices_to_check:

            closest_point_t = np.dot(self.graph.nodes[u]['pos'] - v1, direction) / np.dot(direction, direction)
            closest_point = v1 + closest_point_t * direction
            if 0 <= closest_point_t <= l and np.linalg.norm(closest_point - self.graph.nodes[u]['pos']) <= d:
                vertices_within_rectangle.add(u)

        return vertices_within_rectangle

    def edge_edge_unsafe_interval(self, e1, e2):
        """ Calculates the unsafe interval for an agent to traverse e2 when e1 is traversed by another agent at time 0.

        Uses "5.1 Exact Delay for Constant Velocity" from Walker & Sturtenvant 2019, but modified since those equations
        do not seem to be correct.

        Args:
            e1: edge label
            e2: edge label

        Returns:
            unsafe_interval: list of two floats, the unsafe interval for the agent to traverse e2. If no overlap, None.
        """

        # Get the positions of the edge vertices, and the velocities of each agent along its edge
        v11, v12 = self.edge_coords(e1)
        d1 = np.linalg.norm(v12 - v11) * self.s
        V1 = (v12 - v11) / np.linalg.norm(v12 - v11) * self.s

        v21, v22 = self.edge_coords(e2)
        d2 = np.linalg.norm(v22 - v21) * self.s
        V2 = (v22 - v21) / np.linalg.norm(v22 - v21) * self.s

        # Corner points of the possible-collision-region
        fregion_p1 = (0, 0)
        fregion_p2 = (d1, d1)
        fregion_p3 = (d1, d1 - d2)
        fregion_p4 = (0, -d2)

        # collision conic section constants
        A = np.dot(V1 - V2, V1 - V2)
        B = 2 * np.dot(V2, V1 - V2)
        C = np.dot(V2, V2)
        D = 2 * np.dot(v11 - v21, V1 - V2)
        E = 2 * np.dot(V2, v11 - v21)
        F = np.dot(v11 - v21, v11 - v21) - (self.r + self.r) ** 2

        # special case: the velocities are the same
        if np.linalg.norm(V1 - V2) < 1e-6:

            p = -E / C
            q = F / C
            if (p ** 2) / 4 - q >= 0:
                delta1 = -p / 2 - np.sqrt((p ** 2) / 4 - q)
                delta2 = -p / 2 + np.sqrt((p ** 2) / 4 - q)

                # Check for overlap
                if delta1 <= max(fregion_p1[1], fregion_p2[1]) and min(fregion_p3[1], fregion_p4[1]) <= delta2:
                    return [max(delta1, min(fregion_p3[1], fregion_p4[1])),
                            min(delta2, max(fregion_p1[1], fregion_p2[1]))]
                else:
                    return None
            else:
                return None

        # Get all ellipse/boundary intersection points
        points_in_region = []

        # --- left boundary t=0
        p = E / C
        q = F / C
        if (p ** 2) / 4 - q >= 0:
            for delta in [-p / 2 - np.sqrt((p ** 2) / 4 - q), -p / 2 + np.sqrt((p ** 2) / 4 - q)]:
                if fregion_p4[1] <= delta <= fregion_p1[1]:
                    points_in_region.append((0, delta))

        # --- right boundary t = d1
        p = (B * d1 + E) / C
        q = (A * d1 ** 2 + D * d1 + F) / C
        if (p ** 2) / 4 - q >= 0:
            for delta in [-p / 2 - np.sqrt((p ** 2) / 4 - q), -p / 2 + np.sqrt((p ** 2) / 4 - q)]:
                if fregion_p3[1] <= delta <= fregion_p2[1]:
                    points_in_region.append((d1, delta))

        # --- upper boundary delta = t
        p = (D + E) / (A + B + C)
        q = F / (A + B + C)
        if (p ** 2) / 4 - q >= 0:
            for delta in [-p / 2 - np.sqrt((p ** 2) / 4 - q), -p / 2 + np.sqrt((p ** 2) / 4 - q)]:
                if fregion_p1[0] <= delta <= fregion_p2[0]:
                    points_in_region.append((delta, delta))

        # --- lower boundary delta = t - d2
        p = (2 * A * d2 + B * d2 + D + E) / (A + B + C)
        q = (A * d2 ** 2 + D * d2 + F) / (A + B + C)
        if (p ** 2) / 4 - q >= 0:
            for delta in [-p / 2 - np.sqrt((p ** 2) / 4 - q), -p / 2 + np.sqrt((p ** 2) / 4 - q)]:
                if fregion_p4[0] <= delta + d2 <= fregion_p3[0]:
                    points_in_region.append((delta + d2, delta))

        # Get extrema points (only if the conic section is an ellipse)
        if B ** 2 - 4 * A * C < 0:
            c = (B * D - 2 * A * E) / (4 * A * C - B ** 2)
            h = np.sqrt((2 * B * D - 4 * A * E) ** 2 + 4 * (4 * A * C - B ** 2) * (D ** 2 - 4 * A * F)) / (
                        2 * (4 * A * C - B ** 2))
            for delta in [c - h, c + h]:
                t = (-B * delta - D) / (2 * A)
                if fregion_p1[0] <= t <= fregion_p2[0] and t - d2 <= delta <= t:
                    points_in_region.append((t, delta))

        # Check if any corner points are within the collision region
        for (t, delta) in [fregion_p1, fregion_p2, fregion_p3, fregion_p4]:
            if A * t ** 2 + B * t * delta + C * delta ** 2 + D * t + E * delta + F <= 0:
                points_in_region.append((t, delta))

        # Get the unsafe interval
        if not points_in_region:
            return None

        min_delta = min(p[1] for p in points_in_region)
        max_delta = max(p[1] for p in points_in_region)
        unsafe_interval = [min_delta, max_delta]

        return unsafe_interval

    def vertex_edge_unsafe_interval(self, v, e):
        """Calculates the unsafe interval for an agent to traverse edge e when another agent is at vertex v.

        Args:
            v: Vertex where a stationary agent is located at time 0.
            e: Edge being traversed by a moving agent.

        Returns:
            A list [t_start, t_end] representing the unsafe time interval if a
            collision is possible, or None if no collision is possible.

        Notes:
            Time is normalized such that the moving agent reaches the end of edge e at
            time 0 and starts at time -d2.
        """

        v11 = np.array(self.graph.nodes[v]['pos'])
        v21, v22 = self.edge_coords(e)
        d2 = np.linalg.norm(v22 - v21) * self.s
        V2 = (v22 - v21) / np.linalg.norm(v22 - v21) * self.s

        # min/max points of the possible-collision-region
        p1 = 0
        p4 = -d2

        p = 2 * np.dot(V2, v11 - v21) / np.dot(V2, V2)
        q = (np.dot(v11 - v21, v11 - v21) - (self.r + self.r) ** 2) / np.dot(V2, V2)
        if (p ** 2) / 4 - q >= 0:
            delta1 = -p / 2 - np.sqrt((p ** 2) / 4 - q)
            delta2 = -p / 2 + np.sqrt((p ** 2) / 4 - q)
            if delta1 <= p1 and p4 <= delta2:
                return [max(delta1, p4), min(delta2, p1)]

        return None
