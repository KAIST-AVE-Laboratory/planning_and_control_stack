import rospy

from ave_planning_stack.behavioral_agent import BehevioralAgent
from ave_planning_stack.traffic_oracle import TrafficOracle


if __name__ == "__main__":
    rospy.init_node("behavioral_agent_node")

    try:
        ## get rosparam
        role_name = rospy.get_param("~/role_name", "ego_vehicle")
        nominal_speed = rospy.get_param("~/nominal_speed", 20.)

        ## create the behavioral agent
        traffic_sensor = TrafficOracle(role_name)
        behavioral_agent = BehevioralAgent(traffic_sensor, nominal_speed)
        
        ## setup safety stopping
        rospy.on_shutdown(behavioral_agent.emergency_stop)

        ## setup behavioral agent frequency
        rospy.Timer(rospy.Duration(0.05), lambda timer_event: behavioral_agent.run_step())

        rospy.loginfo("%s node is running." % rospy.get_name())
        rospy.spin()
    
    except (rospy.ROSInterruptException, rospy.ROSException) as e:
        if not rospy.is_shutdown():
            rospy.logwarn("ROS Error during execution: {}".format(e))
            
    except KeyboardInterrupt:
        rospy.loginfo("User requested shut down.")