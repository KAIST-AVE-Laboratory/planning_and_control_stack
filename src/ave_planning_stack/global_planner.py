import rospy

from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Odometry, Path

from ave_planning_stack.graph_map import GraphMap

class GlobalPlanner:
    """Global path planner."""

    def __init__(self, role_name: str, graph_map: GraphMap):
        """Constructor."""
        self.role_name = role_name
        self.graph_map = graph_map

        ## variables for subscriber callbacks
        self.goal_location = None
        self.ego_vehicle_pose = None

        ## ros subscribers
        self._goal_subscriber = rospy.Subscriber(
            "/carla/%s/goal" % self.role_name,
            Point,
            self._goal_callback
        )

        self._odometry_subscriber = rospy.Subscriber(
            "/carla/%s/odometry" % self.role_name,
            Odometry,
            self._odometry_callback
        )

        ## ros publishers
        self._waypoints_publisher = rospy.Publisher(
            "/carla/%s/waypoints" % self.role_name,
            Path,
            queue_size=10
        )
    

    def _goal_callback(self, msg: Point) -> None:
        """"""
        self.goal_location = (msg.x, msg.y)
        
        ## path search
        start_location = (self.ego_vehicle_pose.position.x, self.ego_vehicle_pose.position.y)
        list_of_wp = self.graph_map.get_global_route(start_location, self.goal_location)

        ## List[Tuple[float, float]] -> List[PoseStamped]
        path = []
        for wp_idx, wp in enumerate(list_of_wp):
            msg = PoseStamped()
            msg.header.seq = wp_idx
            msg.pose.position.x = wp[0]
            msg.pose.position.y = wp[1]
            path.append(msg)
        
        ## publish
        msg = Path()
        msg.poses = path
        self._waypoints_publisher.publish(msg)


    def _odometry_callback(self, msg: Odometry) -> None:
        """"""
        self.ego_vehicle_pose = msg.pose.pose