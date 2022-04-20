import rospy
import sys

from carla_msgs.msg import CarlaEgoVehicleInfo
from nav_msgs.msg import Odometry

from planning_and_control_stack.msg import Behavior
from ave_planning_stack.traffic_oracle import TrafficOracle

class BehevioralAgent:
    """Implements the agent that acts on behavioral layer."""

    def __init__(self, traffic_sensor: TrafficOracle, nominal_speed: float):
        """Constructor.
        
        """
        self.traffic_sensor = traffic_sensor
        self.role_name = self.traffic_sensor.role_name
        self.nominal_speed = nominal_speed

        ## wait for ego vehicle
        try:
            rospy.loginfo("Waiting for ego vehicle indefinitely...")
            self.vehicle_info = rospy.wait_for_message(
                "/carla/%s/vehicle_info" % self.role_name,
                CarlaEgoVehicleInfo
            )
            self.traffic_sensor.ego_vehicle_id = self.vehicle_info.id
            rospy.loginfo("Ego vehicle found!")

        except rospy.ROSException:
            rospy.logerr("Timeout while waiting for ego vehicle!")
            sys.exit(1)

        ## variables for subscriber callbacks
        self.ego_vehicle_pose = None
        
        ## ros subscribers
        self._odometry_subcsriber = rospy.Subscriber(
            "/carla/%s/odometry" % self.role_name,
            Odometry,
            self._odometry_callback
        )

        ## ros publishers
        self._behavior_publisher = rospy.Publisher(
            "/carla/%s/behavior" % self.role_name,
            Behavior,
            queue_size=10
        )
    

    def _odometry_callback(self, msg: Odometry):
        """"""
        self.ego_vehicle_pose = msg.pose.pose
    

    def emergency_stop(self):
        """To be called on shutdown."""
        msg = Behavior()
        msg.behavior = Behavior.EMERGENCY_STOP
        self._behavior_publisher.publish(msg)


    def run_step(self):
        """"""
        ## in the `ad_agent.py` of `carla_ad_agent` package, they do deep copy
        ## of the variables and use `threading.lock` to make sure the access is
        ## thread-safe. i don't see the need to do that right now
        ego_vehicle_pose = self.ego_vehicle_pose
        objects = self.traffic_sensor.objects
        tl_infos = self.traffic_sensor.traffic_lights_info
        tl_status = self.traffic_sensor.traffic_lights_status

        ## check initial condition
        if ego_vehicle_pose is None:
            rospy.loginfo("Waiting for ego vehicle pose.")
            return
        
        if set(tl_infos.keys()) != set(tl_status.keys()):
            rospy.logwarn("Missing traffic light information.")
            return
        
        ## check the traffic
        is_obstacle_detected, obstacle_velocity = self.traffic_sensor.is_vehicle_hazard(ego_vehicle_pose, objects)
        is_red_light, stopping_distance = self.traffic_sensor.is_red_light(ego_vehicle_pose, tl_status, tl_infos)
        
        ## decides the behavior
        msg = Behavior()
        msg.time = rospy.Time.now()

        if is_obstacle_detected:
            msg.behavior = Behavior.CAR_FOLLOWING
            msg.data = obstacle_velocity

        elif is_red_light:
            msg.behavior = Behavior.RED_LIGHT_STOP
            msg.data = stopping_distance

        else:
            msg.behavior = Behavior.NOMINAL
            msg.data = self.nominal_speed
        
        ## publish
        self._behavior_publisher.publish(msg)