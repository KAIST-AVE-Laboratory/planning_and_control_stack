from typing import Tuple

import math
import numpy as np
import rospy
import scipy.interpolate as interp
import scipy.spatial.distance as dist

from carla_msgs.msg import CarlaEgoVehicleControl
from nav_msgs.msg import Odometry, Path
from tf.transformations import euler_from_quaternion

from planning_and_control_stack.msg import Behavior, Obstacles
from ave_control_stack.vehicle_pid_controller import VehiclePIDController
from ave_planning_stack.frenet_planner import FrenetPlanner

class LocalPlanner:

    def __init__(self,
        role_name: str,
        # control_timestep: float,
        nominal_speed: float,
        wp_closeness: float,
        wp_distance_threshold: float,
        # planning_lookahed_distance: float,

        ## PID controller params
        kp_lateral: float, ki_lateral: float, kd_lateral: float,
        kp_longitudinal: float, ki_longitudinal: float, kd_longitudinal: float,

        ## Frenet planner params
        dt: float, n_dt: int,
        dd: float, n_dd: int,
        kj: float, kt: float, kd: float,
        k_lat: float, k_lon: float,
        ds: float, n_ds: int,
        d0: float, tw: float,
        ds_dot: float, n_ds_dot: int) -> None:
        """"""
        assert wp_distance_threshold > 0.
        # assert planning_lookahed_distance > 0.

        self.role_name = role_name
        # self.control_timestep = control_timestep
        self.nominal_speed = nominal_speed
        self.wp_closeness = wp_closeness
        self.wp_distance_threshold = wp_distance_threshold
        # self.planning_lookahed_distance = planning_lookahed_distance
        self.ds = ds
        self.n_ds = n_ds
        self.d0 = d0
        self.tw = tw
        self.ds_dot = ds_dot
        self.n_ds_dot = n_ds_dot

        lateral_gains = {'K_P': kp_lateral, 'K_D': kd_lateral, 'K_I': ki_lateral}
        longitudinal_gains = {'K_P': kp_longitudinal, 'K_D': kd_longitudinal, 'K_I': ki_longitudinal}
        self.controller = VehiclePIDController(None, lateral_gains, longitudinal_gains)
        self.planner = FrenetPlanner(dt, n_dt, dd, n_dd, kj, kt, kd, k_lat, k_lon)

        ## variables for subscriber callbacks
        self.ego_vehicle_pose = None
        self.ego_vehicle_speed = None
        self.x_spline = None
        self.y_spline = None
        self.spline_samples = None
        self.behavior = None
        self.behavior_data = None
        self.obstacles = None

        ## ros subscribers
        self._odometry_subscriber = rospy.Subscriber(
            "/carla/%s/odometry" % self.role_name,
            Odometry,
            self._odometry_callback
        )

        self._waypoints_subscriber = rospy.Subscriber(
            "/carla/%s/waypoints" % self.role_name,
            Path,
            self._waypoints_callback
        )

        self._behavior_subscriber = rospy.Subscriber(
            "/carla/%s/behavior" % self.role_name,
            Behavior,
            self._behavior_callback
        )

        self._obstacles_subscriber = rospy.Subscriber(
            "/carla/%s/obstacles" % self.role_name,
            Obstacles,
            self._obstacle_callback
        )

        ## ros publishers
        self._vehicle_cmd_publisher = rospy.Publisher(
            "/carla/%s/vehicle_control_cmd" % self.role_name,
            CarlaEgoVehicleControl,
            queue_size=10
        )

        # self._path_publisher = 


    def _odometry_callback(self, msg: Odometry):
        """"""
        ## get the (x, y, yaw)
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        rot = msg.pose.pose.orientation
        quaternion = [rot.x, rot.y, rot.z, rot.w]
        _, _, yaw = euler_from_quaternion(quaternion)

        ## save the values
        self.ego_vehicle_pose = (x, y, yaw)
        self.ego_vehicle_speed = math.sqrt(msg.twist.twist.linear.x ** 2 +
                                           msg.twist.twist.linear.y ** 2 +
                                           msg.twist.twist.linear.z ** 2) * 3.6
    

    def _waypoints_callback(self, msg: Path):
        """"""
        ## store the xs, ys, and ds of the waypoints where d is the distance to
        ## the first waypoint along the path ie. the s-axis of the frenet frame
        xs = [msg.poses[0].pose.position.x]
        ys = [msg.poses[0].pose.position.y]
        ds = [0]
        distance = 0 ## container to accumulate distance between waypoints
        
        for idx, (pose_prv, pose_now) in enumerate(zip(msg.poses[:-1], msg.poses[1:])):
            x_now, y_now = pose_now.pose.position.x, pose_now.pose.position.y
            x_prv, y_prv = pose_prv.pose.position.x, pose_prv.pose.position.y
            distance += math.sqrt((x_now - x_prv)**2 + (y_now - y_prv)**2)

            ## skip the waypoint if it is below threshold and not the last waypoint
            if distance < self.wp_distance_threshold and idx != (len(msg.poses) - 2):
                continue
            
            xs.append(x_now)
            ys.append(y_now)
            ds.append(distance + ds[-1])
            distance = 0
        
        rospy.loginfo("Receiving %d waypoints separated at least %.3f m with a total length of %.3f m."
            % (len(xs), self.wp_distance_threshold, ds[-1]))

        ## fit cubic spline for this waypoints
        self.x_spline = interp.CubicSpline(ds, xs, bc_type=((1, self.ego_vehicle_speed), (1, 0.)))
        self.y_spline = interp.CubicSpline(ds, ys, bc_type=((1, self.ego_vehicle_speed), (1, 0.)))

        ## sample the splines
        ss = np.linspace(ds[0], ds[-1], int(ds[-1]/self.wp_closeness))
        x_spline_samples = self.x_spline(ss)
        y_spline_samples = self.y_spline(ss)
        self.spline_samples = np.stack((x_spline_samples, y_spline_samples), axis=-1)

        rospy.loginfo("Successfully fitting the splines!")
    
    
    def _behavior_callback(self, msg: Behavior):
        """"""
        self.behavior = msg.behavior
        self.behavior_data = msg.data
    

    def _obstacle_callback(self, msg: Obstacles):
        """"""
        self.obstacles = msg
    

    def is_circle_colliding(self,
        c1: Tuple[float, float, float],
        c2: Tuple[float, float, float]) -> bool:
        """"""
        d = np.linalg.norm([c1[0] - c2[0], c1[1] - c2[1]])
        r = c1[2] + c2[2]
        if d > r: return False
        return True
    

    def is_collision(self, x: float, y: float, r: float) -> bool:
        """"""
        if self.obstacles is not None:
            for x_, y_, r_ in zip(self.obstacles.x, self.obstacles.y, self.obstacles.r):
                if self.is_circle_colliding((x, y, r), (x_, y_, r_)): return True
        return False
    

    def emergency_stop(self):
        """"""
        control_msg = CarlaEgoVehicleControl()
        control_msg.steer = 0.0
        control_msg.throttle = 0.0
        control_msg.brake = 1.0
        control_msg.hand_brake = False
        control_msg.manual_gear_shift = False
        self._vehicle_cmd_publisher.publish(control_msg)
    

    def world2frenet(self, pose, speed):
        """Convert the given (x, y, yaw, v) to (s, s_dot, d, d_dot)."""
        ego_x, ego_y, ego_yaw = pose

        ## find the ego closest s coordinates on the spline
        current_position = np.array([[ego_x, ego_y]])
        distance_to_spline = dist.cdist(self.spline_samples, current_position)
        closest_idx = np.argmin(distance_to_spline)
        closest_dist = distance_to_spline[closest_idx, 0]
        s = closest_idx * self.wp_closeness

        ## get the (x, y, yaw_road[ie. road_direction]) of the closest s
        ## first derivative is ill-defined at the boundary (start point of the spline)
        x  = self.x_spline(s, 0)
        y  = self.y_spline(s, 0)
        dx = self.x_spline(s + 1e-6, 1)
        dy = self.y_spline(s + 1e-6, 1)
        yaw_road = math.atan2(dy, dx)

        ## get the frenet frame basis vector in world frame rooted at closest s with ego
        ## disp_x = ego_x - x
        ## disp_y = ego_y - y
        ## disp_d = np.linalg.norm([disp_x, disp_y])
        ## d_unit_vector = np.array([disp_x/disp_d, disp_y/disp_d])
        ## s_unit_vector = np.array([math.cos(yaw_road), math.sin(yaw_road)])
        
        ## get the d coordinates (and d_dot) on the closest point on spline
        d = math.sqrt((ego_x - x)**2 + (ego_y - y)**2)

        ## calculate the vehicle velocity in frenet frame
        ## yaw_wrt_s means yaw with respect to s-axis of frenet
        yaw_wrt_s = ego_yaw - yaw_road
        s_dot = speed * math.cos(yaw_wrt_s)
        d_dot = speed * math.sin(yaw_wrt_s)

        return s, s_dot, d, d_dot
    

    def frenet2world(self, s, s_dot, d, d_dot):
        """Convert the given (s, s_dot, d, d_dot) to (x, y, yaw, v)."""
        ## get the (v)
        v = np.linalg.norm([s_dot, d_dot])

        ## get yaw of the road
        ## first derivative is ill-defined at the boundary (start point of the spline)
        dx = self.x_spline(s + 1e-6, 1)
        dy = self.y_spline(s + 1e-6, 1)
        yaw_road = math.atan2(dy, dx)

        ## get the (x, y)
        x = self.x_spline(s, 0) + math.cos(yaw_road + math.pi/2) * d
        y = self.y_spline(s, 0) + math.sin(yaw_road + math.pi/2) * d

        ## get the (yaw)
        ## yaw_wrt_s means yaw with respect to s-axis of frenet
        yaw_wrt_s = math.atan2(d_dot, s_dot)
        yaw = yaw_wrt_s + yaw_road

        return x, y, yaw, v
    

    def run_step(self):
        """"""
        if self.spline_samples is None:
            rospy.loginfo("Waiting for a route...")
            self.emergency_stop()
            return
        
        ## ------------------- EMERGENCY STOP -------------------

        if self.behavior == Behavior.EMERGENCY_STOP:
            self.emergency_stop()
            return

        ## ------------------- NOMINAL -------------------

        elif self.behavior == Behavior.NOMINAL:
            s, s_dot, d, d_dot = self.world2frenet(self.ego_vehicle_pose, self.ego_vehicle_speed)

            ## for now assume that ego moves at constant speed
            c_d = (d, d_dot, 0.)
            c_s = (s, s_dot, 0.)
            t_s = (self.nominal_speed, 0.)

            ## generate the possible paths in frenet frame
            param = (c_d, c_s, t_s, self.ds_dot, self.n_ds_dot)
            paths = self.planner.velocity_tracking(*param)

            ## sort the paths based on cost
            _, paths = zip(*sorted(zip(paths["cost"], paths["path"])))

            ## find the first path that is collision-free
            ## but first, need to convert the best path to world frame
            for d_path, d_dot_path, _, s_path, s_dot_path, _ in paths:
                free_path = True
                
                ## check colissiong along a path, if collides: immediately break
                for d, d_dot, s, s_dot in zip(d_path, d_dot_path, s_path, s_dot_path):
                    x, y, yaw, v = self.frenet2world(s, s_dot, d, d_dot)
                    
                    if self.is_collision(x, y, 2.):
                        free_path = False
                        break
                
                ## if it is free path, get the (d, d_dot, s, s_dot)
                if free_path:
                    d, d_dot = d_path[2], d_dot_path[2]
                    s, s_dot = s_path[2], s_dot_path[2]
                    break
        
        ## ------------------- RED LIGHT STOP -------------------

        elif self.behavior == Behavior.RED_LIGHT_STOP:
            s, s_dot, d, d_dot = self.world2frenet(self.ego_vehicle_pose, self.ego_vehicle_speed)

            ## for now assume that ego moves at constant speed
            c_d = (d, d_dot, 0.)
            c_s = (s, s_dot, 0.)
            t_s = (self.behavior_data, 0., 0.)

            ## generate the possible paths in frenet frame
            ## d0 and tw is 0 because we dont want to maintain time-gap as in the car folowing mode
            param = (c_d, c_s, t_s, self.ds, self.n_ds, 0., 0.)
            paths = self.planner.plan_distance_keeping(*param)

            ## sort the paths based on cost
            _, paths = zip(*sorted(zip(paths["cost"], paths["path"])))

            ## find the first path that is collision-free
            ## but first, need to convert the best path to world frame
            for d_path, d_dot_path, _, s_path, s_dot_path, _ in paths:
                free_path = True
                
                ## check colissiong along a path, if collides: immediately break
                for d, d_dot, s, s_dot in zip(d_path, d_dot_path, s_path, s_dot_path):
                    x, y, yaw, v = self.frenet2world(s, s_dot, d, d_dot)
                    
                    if self.is_collision(x, y, 2.):
                        free_path = False
                        break
                
                ## if it is free path, get the (d, d_dot, s, s_dot)
                ## the "2" is just arbitrary
                if free_path:
                    d, d_dot = d_path[2], d_dot_path[2]
                    s, s_dot = s_path[2], s_dot_path[2]
                    break
        
        ## ------------------- EXECUTE -------------------

        x, y, yaw, v = self.frenet2world(s, s_dot, d, d_dot)
        control_msg = self.controller.run_step(20,
            self.ego_vehicle_speed, self.ego_vehicle_pose, (x, y))
        
        self._vehicle_cmd_publisher.publish(control_msg)
        