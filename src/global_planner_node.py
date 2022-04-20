#!/usr/bin/env python

"""
Generates a plan of waypoints to follow.

Subscribes:
- /carla/
"""

import rospkg
import rospy
import os.path as osp

from ave_planning_stack.global_planner import GlobalPlanner
from ave_planning_stack.graph_map import GraphMap


if __name__ == "__main__":
    rospy.init_node("global_planner_node")

    try:
        ## get the path to planning_and_control_stack
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path("planning_and_control_stack")

        ## get rosparam
        role_name = rospy.get_param("~/role_name", "ego_vehicle")
        map_filepath = rospy.get_param("~/map_filepath", osp.join(pkg_path, "dataset/Town01_graphmap.pickle"))

        ## load the graph map
        if map_filepath.split(".")[-1] == "pickle":
            graph_map = GraphMap.from_pickle(map_filepath)
        elif map_filepath.split(".")[-1] == "shp":
            graph_map = GraphMap.from_shapefile(map_filepath)
        
        ## spin the global planner
        global_planner = GlobalPlanner(role_name, graph_map)
        rospy.loginfo("%s node is running." % rospy.get_name())
        rospy.spin()
    
    except (rospy.ROSInterruptException, rospy.ROSException) as e:
        if not rospy.is_shutdown():
            rospy.logwarn("ROS Error during execution: {}".format(e))
            
    except KeyboardInterrupt:
        rospy.loginfo("User requested shut down.")