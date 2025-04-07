import numpy as np
from time import time
import matplotlib.pyplot as plt
import datetime


def edge_edge_intersection(e1, e2):
    """ Checks if two lines intersect.

    Args:
        e1: 2x2 numpy array of end-point coordinates
        e2: 2x2 numpy array of end-point coordinates

    Returns:
        bool, True if the lines intersect.
    """

    # get lengths and directions
    l1 = np.linalg.norm(e1[1] - e1[0])
    l2 = np.linalg.norm(e2[1] - e2[0])
    d1 = (e1[1] - e1[0]) / l1
    d2 = (e2[1] - e2[0]) / l2

    # check if parallel, collinear, and in that case overlapping
    if np.cross(d1, d2) == 0:

        # check if collinear
        if np.cross(e2[0] - e1[0], d1) == 0:

            # check if overlapping
            if 0 <= np.dot(e2[0] - e1[0], d1) <= l1 or 0 <= np.dot(e2[1] - e1[0], d1) <= l1:
                return True
            if 0 <= np.dot(e1[0] - e2[0], d2) <= l2 or 0 <= np.dot(e1[1] - e2[0], d2) <= l2:
                return True
        else:
            return False

    # check if they intersect
    t1 = np.cross(e2[0] - e1[0], d2) / np.cross(d1, d2)
    t2 = np.cross(e1[0] - e2[0], d1) / np.cross(d2, d1)

    if 0 <= t1 <= l1 and 0 <= t2 <= l2:
        return True
    else:
        return False


def get_overlapping_edges(edges, min_box_factor):
    """ Experimental method for finding overlapping edges.

    Args:
        edges: dict of edges, keys: edge labels, values: edge end-point coordinates (numpy array).
    """

    def edge_intersects_box(e, box):
        """ Checks if an edge intersects a box.

        If the edge touches only the boundary, such as if an end-point is on the boundary, then there is only an
        intersection on the left and lower boundaries.

        Args:
            e: 2x2 numpy array of edge end-points.
            box: 2x2 numpy array of box lower-left and upper-right coordinates.

        Returns:
            intersects: bool, True if the edge intersects the box.
        """
        # Initial check 1: are they fully above, below, left or right of each other?
        if max(e[0][0], e[1][0]) < box[0][0] or min(e[0][0], e[1][0]) >= box[1][0]:
            return False
        if max(e[0][1], e[1][1]) < box[0][1] or min(e[0][1], e[1][1]) >= box[1][1]:
            return False

        # Initial check 2: are any of the end-points inside the box?
        if box[0][0] <= e[0][0] < box[1][0] and box[0][1] <= e[0][1] < box[1][1]:
            return True
        if box[0][0] <= e[1][0] < box[1][0] and box[0][1] <= e[1][1] < box[1][1]:
            return True

        # Check if the edge intersects the box
        box_edges = [
            np.array((box[0], (box[0][0], box[1][1]))),  # lower-left to upper-left
            np.array((box[0], (box[1][0], box[0][1]))),  # lower-left to lower-right
            np.array((box[1], (box[1][0], box[0][1]))),  # upper-right to lower-right
            np.array((box[1], (box[0][0], box[1][1])))   # upper-right to upper-left
        ]
        for be in box_edges:
            if edge_edge_intersection(e, be):
                return True

    def get_all_edge_intersections(edges):
        """ Checks for intersections between all edges.

        Args:
            edges: dict of edges, keys: edge labels, values: edge end-point coordinates (numpy array).

        Returns:
            set of edge-pair-tuples that intersect.
        """
        checked_edges = set()
        overlapping_pairs = set()
        for label1, e1 in edges.items():

            checked_edges.add(label1)  # doing this now also avoids having an extra check for e1 == e2

            for label2, e2 in edges.items():

                if label2 in checked_edges:
                    continue

                # Initial check: have the edges already been checked before?
                if label2 in checked_intersections[label1]:
                    if label2 in intersections[label1]:
                        overlapping_pairs.add((label1, label2))
                elif edge_edge_intersection(e1, e2):
                    overlapping_pairs.add((label1, label2))

        return overlapping_pairs

    def recursive_split(box, E):
        """

        Args:
            box: 2x2 numpy array of lower-left and upper-right coordinates.
        """
        box_center = ((box[0][0] + box[1][0]) / 2, (box[0][1] + box[1][1]) / 2)
        cells = [
            np.array((box[0], box_center)),                                       # lower-left
            np.array(((box_center[0], box[0][1]), (box[1][0], box_center[1]))),   # lower-right
            np.array((box_center, box[1])),                                       # upper-right
            np.array(((box[0][0], box_center[1]), (box_center[0], box[1][1])))    # upper-left
        ]

        for cell in cells:

            """
            num_cells[0] += 1
            # ##### PLOTTING #####
            ax.plot([cell[0][0], cell[1][0], cell[1][0], cell[0][0], cell[0][0]],
                    [cell[0][1], cell[0][1], cell[1][1], cell[1][1], cell[0][1]],
                    'r-')
            plt.draw()
            ax.set_aspect('equal')
            plt.pause(0.001)  # Pause briefly to update the figure
            fig.canvas.flush_events()
            # ##### PLOTTING #####
            """

            # Check if any edges intersect the cell
            edges_in_box = dict()
            for label, e in E.items():
                if edge_intersects_box(e, cell):
                    edges_in_box[label] = e

            # Remove edges that have already been checked with *all* other edges in the box
            """
            ordered_edges = list(edges_in_box)
            not_removed = set()
            for i in range(len(ordered_edges)):
                e1 = ordered_edges[i]
                if e1 in not_removed:
                    continue

                for j in range(len(ordered_edges)):
                    e2 = ordered_edges[j]
                    if e1 == e2:
                        continue
                    if e2 not in checked_intersections[e1]:
                        not_removed.add(e1)
                        not_removed.add(e2)
                        break
            edges_in_box = {label: edges_in_box[label] for label in not_removed}         
            """

            if len(edges_in_box) <= 1:
                continue

            if len(edges_in_box) == 2 or cell[1][0] - cell[0][0] < min_size or cell[1][1] - cell[0][1] < min_size:
                intersecting_pairs = get_all_edge_intersections(edges_in_box)
                for label1 in edges_in_box:
                    for label2 in edges_in_box:
                        if label1 != label2:
                            checked_intersections[label1].add(label2)
                for e1, e2 in intersecting_pairs:
                    intersections[e1].add(e2)
                    intersections[e2].add(e1)
            else:
                recursive_split(cell, {e: E[e] for e in edges_in_box})

    # #### PLOTTING START ####
    """
    plt.ion()  # Enable interactive mode before creating the figure
    fig, ax = plt.subplots()
    for e in edges.values():
        ax.plot([e[0][0], e[1][0]], [e[0][1], e[1][1]], 'k-')
    plt.draw()
    ax.set_aspect('equal')
    plt.pause(0.001)  # Pause briefly to update the figure
    fig.canvas.flush_events()
    #### PLOTTING END ####
    num_cells = [0]
    """

    intersections = {label: set() for label in edges.keys()}
    checked_intersections = {label: set() for label in edges.keys()}

    min_x = np.inf
    min_y = np.inf
    max_x = -np.inf
    max_y = -np.inf
    for label, e in edges.items():
        min_x = min(min_x, e[0][0], e[1][0])
        min_y = min(min_y, e[0][1], e[1][1])
        max_x = max(max_x, e[0][0], e[1][0])
        max_y = max(max_y, e[0][1], e[1][1])

    box = np.array(((min_x, min_y), (max_x, max_y)))
    min_size = min(max_x - min_x, max_y - min_y) / min_box_factor

    recursive_split(box, edges)

    return intersections


def generate_edge_set(n, max_x, max_y):
    """ Generates a set of random edges.

    Args:
        n: int, number of edges to generate.
        max_x: float, maximum x-coordinate.
        max_y: float, maximum y-coordinate.

    Returns:
        edges: dict, keys: edge labels, values: edge end-point coordinates (numpy array).
    """

    edges = dict()
    for i in range(n):
        e = np.random.rand(2, 2) * np.array([max_x, max_y])
        edges['e{}'.format(i)] = e

    return edges






