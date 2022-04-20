"""This script is to visually inspect the resulting plan from `GraphMap`."""

import os.path as osp
import pandas as pd

import carla

## need to import LaneChange and LaneType because the pickled object contains this class
from ave_planning_stack.graph_map import GraphMap, LaneChange, LaneType


DIR_TO_SAVED_CARLA_GRAPHMAP = "/home/wahyu/repos/ros_noetic_ws/src/navmap_waypoint/dataset"
CARLA_MAP_NAME = "Town03"
PICKLE_FILENAME = osp.join(DIR_TO_SAVED_CARLA_GRAPHMAP, "%s_graph_map.pickle" % CARLA_MAP_NAME)

if __name__ == "__main__":
    ## connect to carla
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.load_world(CARLA_MAP_NAME)

    ## load the graph
    graphmap = GraphMap.from_pickle(PICKLE_FILENAME)

    ## plot the nodes in CARLA
    for _, node in graphmap.graph.nodes.data():
        color = carla.Color(255, 0, 0)
        start = carla.Location(node["vertex"][0], node["vertex"][1], 0)
        end = carla.Location(node["vertex"][0], node["vertex"][1], 0.5)
        world.debug.draw_line(start, end, thickness=0.1, color=color)
    
    ## plot the edge
    for edge in graphmap.graph.edges.data():
        entry_node_id, exit_node_id, data = edge
        entry_loc = data["entry"]
        exit_loc = data["exit"]
        path = [entry_loc] + data["path"] + [exit_loc]

        for start, end in zip(path[:-1], path[1:]):
            color = carla.Color(255, 0, 0)
            start = carla.Location(start[0], start[1], 0.5)
            end = carla.Location(end[0], end[1], 0.5)
            world.debug.draw_line(start, end, thickness=0.1, color=color)
    
    ## plan in the graph
    ## weird example because of the map by carla
    start_loc = (10., 18.) ## or (22., 5.)
    end_loc = (200., 50.)
    
    ## good example
    # start_loc = (10., 20.)
    # end_loc = (200., 50.)
    
    ## good example
    # start_loc = (22., 5.)
    # end_loc = (200., 50.)

    ## plot the start (green) and goal position (blue)
    color = carla.Color(0, 255, 0)
    world.debug.draw_line(carla.Location(start_loc[0], start_loc[1], 0), carla.Location(start_loc[0], start_loc[1], 40), color=color)
    color = carla.Color(0, 0, 255)
    world.debug.draw_line(carla.Location(end_loc[0], end_loc[1], 0), carla.Location(end_loc[0], end_loc[1], 40), color=color)
    
    ## A*
    route = graphmap.get_global_route(start_loc, end_loc)

    ## plot the planned graph
    for start, end in zip(route[:-1], route[1:]):
        color = carla.Color(0, 255, 0)
        start = carla.Location(start[0], start[1], 2)
        end = carla.Location(end[0], end[1], 2)
        world.debug.draw_line(start, end, thickness=0.1, color=color)