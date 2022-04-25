#!/usr/bin/env python

import rospy

from ave_planning_stack.local_planner import LocalPlanner

if __name__ == "__main__":
    rospy.init_node("local_planner_node")

    try:
        ## rosparam: General
        role_name = rospy.get_param("~/role_name", "ego_vehicle")
        # control_timestep = rospy.get_param("~/control_timestep", 0.05)
        nominal_speed = rospy.get_param("~/nominal_speed", 20.)
        waypoint_closeness = rospy.get_param("~/waypoint_closeness", 1)
        waypoint_distance_threshold = rospy.get_param("~/waypoint_distance_threshold", 5)
        # planning_lookahed_distance = rospy.get_param("~/planning_lookahed_distance", 10)

        ## rosparam: PID Controller
        kp_lateral = rospy.get_param("~/kp_lateral", 0.9)
        ki_lateral = rospy.get_param("~/ki_lateral", 0.0)
        kd_lateral = rospy.get_param("~/kd_lateral", 0.0)

        kp_longitudinal = rospy.get_param("~/kp_longitudinal", 0.206)
        ki_longitudinal = rospy.get_param("~/ki_longitudinal", 0.0206)
        kd_longitudinal = rospy.get_param("~/kd_longitudinal", 0.515)

        ## rosparam: Frenet Planner
        frenet_dt   = rospy.get_param("~/frenet_dt", 0.2)
        frenet_n_dt = rospy.get_param("~/frenet_n_dt", 0)
        frenet_dd   = rospy.get_param("~/frenet_dd", 1.0)
        frenet_n_dd = rospy.get_param("~/frenet_n_dd", 6)
        frenet_kj   = rospy.get_param("~/frenet_kj", 1.0)
        frenet_kt   = rospy.get_param("~/frenet_kt", 1.0)
        frenet_kd   = rospy.get_param("~/frenet_kd", 1.0)
        frenet_k_lateral      = rospy.get_param("~/frenet_k_lateral", 1.0)
        frenet_k_longitudinal = rospy.get_param("~/frenet_k_longitudinal", 1.0)
        frenet_ds   = rospy.get_param("~/frenet_ds", 0.5)
        frenet_n_ds = rospy.get_param("~/frenet_n_ds", 0)
        frenet_d0   = rospy.get_param("~/frenet_d0", 4.0)
        frenet_tw   = rospy.get_param("~/frenet_tw", 3.0)
        frenet_ds_dot   = rospy.get_param("~/frenet_ds_dot", 1.0)
        frenet_n_ds_dot = rospy.get_param("~/frenet_n_ds_dot", 0)

        ## create the local planner
        local_planner = LocalPlanner(
            role_name,
            # control_timestep,
            nominal_speed,
            waypoint_closeness,
            waypoint_distance_threshold,
            # planning_lookahead_distance
            kp_lateral,
            ki_lateral,
            kd_lateral,
            kp_longitudinal,
            ki_longitudinal,
            kd_longitudinal,
            frenet_dt, frenet_n_dt,
            frenet_dd, frenet_n_dd,
            frenet_kj, frenet_kt, frenet_kd,
            frenet_k_lateral, frenet_k_longitudinal,
            frenet_ds, frenet_n_ds,
            frenet_d0, frenet_tw,
            frenet_ds_dot, frenet_n_ds_dot
        )
        
        ## run the local planner
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            local_planner.run_step()
            rate.sleep()
    
    except (rospy.ROSInterruptException, rospy.ROSException) as e:
        if not rospy.is_shutdown():
            rospy.logwarn("ROS Error during execution: {}".format(e))
            
    except KeyboardInterrupt:
        rospy.loginfo("User requested shut down.")