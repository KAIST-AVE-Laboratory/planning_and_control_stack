#!/usr/bin/env python

import rospy
import cv2

from ave_visualization_stack.BEVGenerator import BEVGenerator
from ave_visualization_stack.wheel_feedback import WheelFeedback

if __name__ == "__main__":
    rospy.init_node("visualization_node")

    try:
        ## rosparam: General
        role_name = rospy.get_param("~role_name")
        map_name = rospy.get_param("~map_name")
        map_path = rospy.get_param("~map_path")
        bev_height = rospy.get_param("~bev_height")
        bev_width = rospy.get_param("~bev_width")
        hdmap_px_per_meter = rospy.get_param("~hdmap_px_per_meter")
        bev_px_per_meter = rospy.get_param("~bev_px_per_meter")
        

        ## create the bev_generator
        bev_generator = BEVGenerator(
            role_name,
            map_name,
            map_path,
            bev_height,
            bev_width,
            hdmap_px_per_meter,
            bev_px_per_meter
        )

        wheel_name = rospy.get_param("~wheel_name")
        kp = rospy.get_param("~kp")
        ki = rospy.get_param("~ki")
        kd = rospy.get_param("~kd")
        
        # wheel_feedback = WheelFeedback(
        #     role_name,
        #     map_name,
        #     [kp, ki, kd]
        # )
        
        ## run the bev_generator
        rospy.spin()
    
    except (rospy.ROSInterruptException, rospy.ROSException) as e:
        if not rospy.is_shutdown():
            rospy.logwarn("ROS Error during execution: {}".format(e))
            
    except KeyboardInterrupt:
        rospy.loginfo("User requested shut down.")
