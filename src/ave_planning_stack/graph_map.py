from enum import Enum
from typing import List, Tuple

import networkx as nx
import numpy as np
import scipy.spatial.distance as scipydist


class LaneChange(Enum):
    """Enumerator to identify lane change availability."""
    NONE = 0
    RIGHT = 1
    LEFT = 2
    BOTH = 3
    UNDETERMINED = 4


class LaneType(Enum):
    """"""
    STRAIGHT = 1
    CURVE = 2
    UNDETERMINED = 3


class GraphMap:
    """Directed graph representation of the map."""

    def __init__(self, graph: nx.DiGraph) -> None:
        """Constructor.
        
        Arguments:
            `graph`: A graph representation of the map.
        """
        self.graph = graph


    def get_global_route(self, start: Tuple[float, float], goal: Tuple[float, float]) -> List[Tuple[float, float]]:
        """"""
        start_nodeid, start_segment = self._localize(start, is_goal=False)
        goal_nodeid, goal_segment = self._localize(goal, is_goal=True)

        ## A* search on the graph
        route = nx.astar_path(self.graph,
            start_nodeid, goal_nodeid,
            self._distance_heuristic, weight="length"
        )

        ## unravel the optimal route/edges into list of coordinates
        waypoints = []
        for n1, n2 in zip(route[:-1], route[1:]):
            edge_data = self.graph.get_edge_data(n1, n2)
            waypoints.append(edge_data["entry"])
            waypoints.extend(edge_data["path"])

        return start_segment + waypoints + goal_segment
    

    def _distance_heuristic(self, nodeid_1: int, nodeid_2: int) -> float:
        """"""
        loc1 = np.array(self.graph.nodes[nodeid_1]["vertex"])
        loc2 = np.array(self.graph.nodes[nodeid_2]["vertex"])
        return np.linalg.norm(loc1 - loc2)
    

    def _localize(self, location: Tuple[float, float], is_goal: bool) -> int:
        """"""
        nearest_distance = float("inf")
        nearest_node_idx = None

        ## search the nearest node in cartesian space
        for node_idx, node_data in self.graph.nodes.data():
            distance = np.linalg.norm(np.array(location) - np.array(node_data["vertex"]))

            if distance < nearest_distance:
                nearest_distance = distance
                nearest_node_idx = node_idx
        
        ## check the neighbors to find the more detailed nearest location on the edge["path"]
        nearest_distance = float("inf")
        nearest_edge = (None, None)
        path_idx_of_the_segment = None

        for succ_node_idx  in self.graph.successors(nearest_node_idx):
            edge_data = self.graph.get_edge_data(nearest_node_idx, succ_node_idx)
            edge_path = [edge_data["entry"]] + edge_data["path"] + [edge_data["exit"]]
            edge_dist = scipydist.cdist([location], edge_path)

            if np.min(edge_dist) <= nearest_distance:
                nearest_distance = np.min(edge_dist)
                nearest_edge = (nearest_node_idx, succ_node_idx)
                path_idx_of_the_segment = np.argmin(edge_dist)
        
        for pred_node_idx  in self.graph.predecessors(nearest_node_idx):
            edge_data = self.graph.get_edge_data(pred_node_idx, nearest_node_idx)
            edge_path = [edge_data["entry"]] + edge_data["path"] + [edge_data["exit"]]
            edge_dist = scipydist.cdist([location], edge_path)

            if np.min(edge_dist) <= nearest_distance:
                nearest_distance = np.min(edge_dist)
                nearest_edge = (pred_node_idx, nearest_node_idx)
                path_idx_of_the_segment = np.argmin(edge_dist)
        
        ## return the nearest node id along with the segment connecting them
        nearest_edge_data = self.graph.get_edge_data(*nearest_edge)

        if is_goal:
            nearest_node_idx = nearest_edge[0]
            connecting_segment = [nearest_edge_data["entry"]] + nearest_edge_data["path"][:path_idx_of_the_segment]
        else:
            nearest_node_idx = nearest_edge[1]
            connecting_segment = nearest_edge_data["path"][path_idx_of_the_segment:]
        
        return nearest_node_idx, connecting_segment
    
    
    @classmethod
    def from_pickle(cls, pickle_filepath: str) -> "GraphMap":
        """Load a pickled `networkx.Digraph` object into a `GraphMap`."""
        import pandas as pd
        graph = pd.read_pickle(pickle_filepath)
        assert isinstance(graph, nx.DiGraph), "The pickled object is not a networkx.Digraph."
        return cls(graph)
    

    @classmethod
    def from_shapefile(cls, shapefile_path: str) -> "GraphMap":
        """"""
        import shapefile
        return cls()