import carla
import math
import rospy

from carla_ad_agent.misc import is_within_distance_ahead
from carla_common import transforms as trans
from carla_msgs.msg import (
    CarlaTrafficLightInfoList,
    CarlaTrafficLightStatus,
    CarlaTrafficLightStatusList
)
from derived_object_msgs.msg import Object, ObjectArray


class TrafficOracle:
    """Oracle agent that provides traffic light and other object status."""

    OBJECT_VEHICLE_CLASSIFICATION = [
        Object.CLASSIFICATION_CAR,
        Object.CLASSIFICATION_BIKE,
        Object.CLASSIFICATION_MOTORCYCLE,
        Object.CLASSIFICATION_TRUCK,
        Object.CLASSIFICATION_OTHER_VEHICLE
    ]

    def __init__(self, role_name: str):
        """Constructor.
        
        """
        self.role_name = role_name
        self.ego_vehicle_id = None ##  will be set by behavioral agent

        ## connect to carla
        self._carla_client = carla.Client("localhost", 2000)
        self._carla_world = self._carla_client.get_world()
        self._carla_map = self._carla_world.get_map()

        ## parameters for hazard detection
        self._proximity_tlight_threshold =  10.0  # meters
        self._proximity_vehicle_threshold = 12.0  # meters

        ## variables for subscriber callbacks
        self.objects = None
        self.traffic_lights_info = None
        self.traffic_lights_status = None
        
        ## ros subscribers
        self._objects_subscriber = rospy.Subscriber(
            "carla/%s/objects" % role_name,
            ObjectArray,
            self._objects_callback
        )

        self._traffic_light_info_subscriber = rospy.Subscriber(
            "/carla/traffic_lights/info",
            CarlaTrafficLightInfoList,
            self._traffic_light_info_callback
        )

        self._traffic_light_status_subscriber = rospy.Subscriber(
            "/carla/traffic_lights/status",
            CarlaTrafficLightStatusList,
            self._traffic_light_status_subscriber
        )
    

    def _objects_callback(self, msg: ObjectArray):
        """Callback for objects subscriber."""
        self.objects = {}
        for obj in msg.objects:
            self.objects[obj.id] = obj
    

    def _traffic_light_status_callback(self, msg: CarlaTrafficLightStatusList):
        """Callback for traffic light status subscriber."""
        self.traffic_lights_status = {}
        for tl_status in msg.traffic_lights:
            self.traffic_lights_status[tl_status.id] = tl_status
    

    def _traffic_light_info_callback(self, msg: CarlaTrafficLightInfoList):
        """Callback for traffic light info subscriber."""
        self.traffic_lights_info = {}
        for tl_info in msg.traffic_lights:
            self.traffic_lights_info[tl_info.id] = tl_info
    

    def _get_trafficlight_trigger_location(self, light_info):
        """"""

        def rotate_point(point, radians):
            """
            rotate a given point by a given angle
            """
            rotated_x = math.cos(radians) * point.x - math.sin(radians) * point.y
            rotated_y = math.sin(radians) * point.x + math.cos(radians) * point.y

            return carla.Vector3D(rotated_x, rotated_y, point.z)

        base_transform = trans.ros_pose_to_carla_transform(light_info.transform)
        base_rot = base_transform.rotation.yaw
        area_loc = base_transform.transform(
            trans.ros_point_to_carla_location(light_info.trigger_volume.center))
        area_ext = light_info.trigger_volume.size

        point = rotate_point(carla.Vector3D(0, 0, area_ext.z / 2.0), math.radians(base_rot))
        point_location = area_loc + carla.Location(x=point.x, y=point.y)

        return carla.Location(point_location.x, point_location.y, point_location.z)
    

    def is_vehicle_hazard(self, ego_vehicle_pose, objects):
        """Checks whether there is a vehicle hazard.

        This method only takes into account vehicles. Pedestrians or other types of obstacles are ignored.

        :param ego_vehicle_pose: current ego vehicle pose
        :type ego_vehicle_pose: geometry_msgs/Pose

        :param objects: list of objects
        :type objects: derived_object_msgs/ObjectArray

        :return: a tuple given by (bool_flag, vehicle), where
                 - bool_flag is True if there is a vehicle ahead blocking us
                   and False otherwise
                 - vehicle is the blocker vehicle id
        """
        ego_vehicle_location = trans.ros_point_to_carla_location(ego_vehicle_pose.position)
        ego_vehicle_waypoint = self._carla_map.get_waypoint(ego_vehicle_location)
        ego_vehicle_transform = trans.ros_pose_to_carla_transform(ego_vehicle_pose)

        for target_vehicle_id, target_vehicle_object in objects.items():
            target_vehicle_location = trans.ros_point_to_carla_location(target_vehicle_object.pose.position)
            target_vehicle_waypoint = self._carla_map.get_waypoint(target_vehicle_location)
            target_vehicle_transform = trans.ros_pose_to_carla_transform(target_vehicle_object.pose)

            ## take into account only vehicles
            if target_vehicle_object.classification not in self.OBJECT_VEHICLE_CLASSIFICATION:
                continue

            ## do not account for the ego vehicle
            if target_vehicle_id == self.ego_vehicle_id:
                continue

            ## can't locate object in the map
            if target_vehicle_waypoint is None:
                continue

            ## if the object is not in our lane it's not an obstacle
            if target_vehicle_waypoint.road_id != ego_vehicle_waypoint.road_id or \
                    target_vehicle_waypoint.lane_id != ego_vehicle_waypoint.lane_id:
                continue

            ## check the distance whether it is in out threshold
            if is_within_distance_ahead(target_vehicle_transform,
                                        ego_vehicle_transform,
                                        self._proximity_vehicle_threshold):
                return (True, target_vehicle_id)

        return (False, None)


    def is_red_light(self, ego_vehicle_pose, lights_status, lights_info):
        """Checks if there is a red light affecting us.
        
        This version of the method is compatible with both European and US style traffic lights.
        
        :param ego_vehicle_pose: current ego vehicle pose
        :type ego_vehicle_pose: geometry_msgs/Pose

        :param lights_status: list containing all traffic light status.
        :type lights_status: carla_msgs/CarlaTrafficLightStatusList

        :param lights_info: list containing all traffic light info.
        :type lights_info: lis.

        :return: a tuple given by (bool_flag, traffic_light), where
                 - bool_flag is True if there is a traffic light in RED
                   affecting us and False otherwise
                 - traffic_light is the traffic light id or None if there is no
                   red traffic light affecting us.
        """
        ego_vehicle_location = trans.ros_point_to_carla_location(ego_vehicle_pose.position)
        ego_vehicle_waypoint = self._carla_map.get_waypoint(ego_vehicle_location)
        ego_vehicle_transform = trans.ros_pose_to_carla_transform(ego_vehicle_pose)

        for light_id in lights_status.keys():
            object_location = self._get_trafficlight_trigger_location(lights_info[light_id])
            object_waypoint = self._carla_map.get_waypoint(object_location)
            object_transform = trans.ros_pose_to_carla_transform(object_waypoint.pose)

            ## can't locate the object in the map
            if object_waypoint is None:
                continue

            ## reduce the search space by ignoring traffic light on different roads
            if object_waypoint.road_id != ego_vehicle_waypoint.road_id:
                continue

            ## reduce the search space by ignoring traffic light
            ## that has different orientation with ego vehicle
            ve_dir = ego_vehicle_transform.get_forward_vector()
            wp_dir = object_transform.get_forward_vector()

            dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z
            if dot_ve_wp < 0: continue

            ## if it is in certain distance, then check the light
            if is_within_distance_ahead(object_transform,
                                        ego_vehicle_transform,
                                        self._proximity_tlight_threshold):
                if lights_status[light_id].state == CarlaTrafficLightStatus.RED or \
                    lights_status[light_id].state == CarlaTrafficLightStatus.YELLOW:
                    return (True, light_id)

        return (False, None)