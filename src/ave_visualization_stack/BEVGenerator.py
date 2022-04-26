import carla
import cv2
import imutils
import math
import numpy as np
import os.path
import rospy

from cv_bridge import CvBridge
from ave_visualization_stack.spline2d import Spline2D

from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Image
from tf.transformations import euler_from_quaternion

class BEVGenerator():
    def __init__(self,
        role_name: str,
        map_name: str,
        map_path: str,
        bev_height: int,
        bev_width: int,
        hdmap_px_per_meter:int ,
        bev_px_per_meter:int) -> None:
        """"""
        hdmap = self.get_hdmap(os.path.join(map_path, 'maps', map_name))
        if hdmap is None: raise ValueError(f'cannot find {map_name}.png')

        self.hdmap_half = hdmap.shape[0]//2
        self.bev_height = bev_height
        self.bev_width = bev_width
        self.bev_diagonal = int(math.sqrt((self.bev_height//2)**2+(self.bev_width//2)**2)) + 10 # 10 for margin
        
        self.hdmap_px_per_meter = hdmap_px_per_meter
        self.bev_px_per_meter = bev_px_per_meter
        self.bev_hdmap_ratio = bev_px_per_meter / hdmap_px_per_meter
        
        self.hdmap = cv2.resize(hdmap[:,:,0], (0,0), fx=self.bev_hdmap_ratio, fy =self.bev_hdmap_ratio)
        self.dot_extent = max(bev_px_per_meter//5,1)

        self.gpmap = None
        self.global_sp = None
        self.local_sp = None
        self.objects_bbox = None
        self.objects_transform = None
        self.ego_transform = None
        self.br = CvBridge()

        self.gp_subscriber = rospy.Subscriber(
            f'/carla/{role_name}/waypoints',
            Path,
            self._gp_callback
        )

        self.lp_subscriber = rospy.Subscriber(
            "/carla/%s/local_path" % role_name,
            Path,
            self._lp_callback
        )
        # self.objects_bbox_subscriber = rospy.Subscriber(
        #     "/carla/.../ob_bbox",
        #     ...,
        #     self._objects_bbox_callback
        # )
        # self.objects_transform_subscriber = rospy.Subscriber(
        #     "/carla/.../ob_transform",
        #     ...,
        #     self._objects_transform_callback
        # )
        self.ego_transform_subscriber = rospy.Subscriber(
            f'/carla/{role_name}/odometry',
            Odometry,
            self._ego_transform_callback
        )
        self._bev_publisher = rospy.Publisher(
            f'/carla/{role_name}/bev',
            Image,
            queue_size=10,
        )
            

    def _gp_callback(self, msg:Path):
        print("Receiving global path...")
        xs = [msg.poses[0].pose.position.x]
        ys = [-msg.poses[0].pose.position.y]

        for pose in msg.poses[1:]:
            x_now, y_now = pose.pose.position.x, pose.pose.position.y
            xs.append(x_now)
            ys.append(-y_now)
        
        global_waypoints = np.stack((xs,ys),axis=0).T

        self.set_sp(global_waypoints, None)

    def _lp_callback(self, msg: Path):
        print("Receiving local path...")
        xs = [msg.poses[0].pose.position.x]
        ys = [-msg.poses[0].pose.position.y]

        for pose in msg.poses[1:]:
            x_now, y_now = pose.pose.position.x, pose.pose.position.y
            xs.append(x_now)
            ys.append(y_now)
        
        local_waypoints = np.stack((xs,ys),axis=0).T

        self.set_sp(None, local_waypoints)

    # def _objects_bbox_callback(self, msg):
    #     self.objects_bbox = ...
    # def _objects_transform_callback(self, msg):
    #     self.objects_transform = ...


    def _ego_transform_callback(self, msg: Odometry):
        # print("got ego transform")
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = 0
        location = carla.Location(x,-y,z)
        rot = msg.pose.pose.orientation
        quaternion = [rot.x, rot.y, rot.z, rot.w]
        _, _, yaw = euler_from_quaternion(quaternion)
        yaw = yaw/math.pi*180*-1
        rotation = carla.Rotation(0, yaw, 0)
        self.ego_transform = carla.Transform(location, rotation)
        self.get_bev()


    def set_sp(self, global_waypoints, local_waypoints) -> None:  # waypoints : N X 2
        if global_waypoints is not None:
            self.global_sp = Spline2D(global_waypoints[:,1], global_waypoints[:, 0])
            self.get_gpmap()
            print("global path drawn!")
        if local_waypoints is not None:
            self.local_sp = Spline2D(local_waypoints[:,1], local_waypoints[:, 0])
            print("local path drawn!")

    def get_hdmap(self, map_name) -> None:
        mtrx = np.float32([[1,0,-1000],[0,1,1000]])
        hdmap = cv2.imread(f'{map_name}.png')
        hdmap_rows,hdmap_cols = hdmap.shape[:2]
        hdmap = cv2.warpAffine(hdmap,mtrx,(hdmap_rows,hdmap_cols),cv2.INTER_LINEAR)

        return hdmap

    def get_gpmap(self) -> None:
        gpmap = np.zeros((self.hdmap_half*2, self.hdmap_half*2, 1))

        u_coords = (self.global_sp.points[:,0] * self.hdmap_px_per_meter + self.hdmap_half).astype(np.int32)
        v_coords = (-self.global_sp.points[:,1] * self.hdmap_px_per_meter + self.hdmap_half).astype(np.int32)
        
        for i in range(len(u_coords)):
            cv2.circle(gpmap, (u_coords[i], v_coords[i]), 3, 255, -1)
        self.gpmap = cv2.resize(gpmap, (0,0), fx=self.bev_hdmap_ratio, fy =self.bev_hdmap_ratio)

    def get_bev(self) -> None:
        ## get the birds-eye-view
        if self.ego_transform is None:
            return np.zeros((self.bev_height, self.bev_width, 3))
        hd_array = self.crop_map(self.hdmap)
        
        ## get the bounding-box
        bb_array = np.zeros((self.bev_height, self.bev_width, 1))
        if (self.objects_bbox is not None) and (self.objects_transform is not None):
            bb_array = self.draw_bbox()
        
        ## get the global path
        gp_array = np.zeros((self.bev_height, self.bev_width, 1))
        if (self.global_sp is not None) and (self.gpmap is not None):
            gp_array = self.crop_map(self.gpmap)
        
        ## get the local path
        if self.local_sp is not None:
            lp_array = self.draw_lp()
            bb_array += lp_array

        bev = np.concatenate((hd_array, gp_array, bb_array),axis=-1)
        self._bev_publisher.publish(self.br.cv2_to_imgmsg(bev.astype(np.uint8), "bgr8"))

    def draw_lp(self):
        lp_array = np.zeros((self.bev_height, self.bev_width, 1))
        points = self.local_sp.points.T
        points = np.r_[
            points, [np.zeros(points.shape[1])]
        ]
        points = np.r_[
            points, [np.ones(points.shape[1])]
        ]
        
        world_2_ego = np.array(self.ego_transform.get_inverse_matrix())
        bev_points = np.dot(world_2_ego, points) * self.bev_px_per_meter

        bev_points = np.array([
            bev_points[1, :] + (self.bev_width//2),
            bev_points[0, :] * -1 + (self.bev_height//2)
        ])

        bev_points = bev_points.T
        points_in_bev_mask = \
            (bev_points[:, 0] > 0.0) & (bev_points[:, 0] < self.bev_width) & \
            (bev_points[:, 1] > 0.0) & (bev_points[:, 1] < self.bev_height)
        bev_points=bev_points[points_in_bev_mask]

        u_coord = bev_points[:, 0].astype(np.int32)
        v_coord = bev_points[:, 1].astype(np.int32)

        for i in range(len(bev_points)):
            cv2.circle(lp_array, (u_coord[i], v_coord[i]), self.dot_extent, 255, -1)
        
        return lp_array

    def draw_bbox(self):
        bb_array = np.zeros((self.bev_height, self.bev_width, 1))

        objects_bbox_cords=[]
        objects_bbox_2_world = []
        objects_world_2_ego = []

        for i in range(len(self.objects_bbox)):
            object_bb = self.objects_bbox[i]
            object_trans = self.objects_transform[i]

            cords = np.zeros((4,4))
            extent = object_bb.extent
            cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
            cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
            cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
            cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
            objects_bbox_cords.append(np.transpose(cords))

            objects_bbox_2_world.append(object_trans.get_matrix())
            objects_world_2_ego.append(self.ego_transform.get_inverse_matrix())
        
        objects_bbox_cords = np.array(objects_bbox_cords)
        objects_bbox_2_world = np.array(objects_bbox_2_world)
        objects_world_2_ego = np.array(objects_world_2_ego)

        objects_world_points = np.matmul(objects_bbox_2_world, objects_bbox_cords)

        objects_bev_points = np.matmul(objects_world_2_ego, objects_world_points) * self.bev_px_per_meter

        objects_bev_points = np.stack([
            objects_bev_points[:, 1, :] + (self.bev_width//2),
            objects_bev_points[:, 0, :] * -1 + (self.bev_height//2)
        ], 1)
        objects_bev_points=np.swapaxes(objects_bev_points,1,2).astype(np.int32)
        
        objects_points_in_bev_mask=\
            (objects_bev_points[:, :, 0] > 0.0) & (objects_bev_points[:, :, 0] < self.bev_width) & \
            (objects_bev_points[:, :, 1] > 0.0) & (objects_bev_points[:, :, 1] < self.bev_height)
        
        objects_points_num_in_bev = np.sum(objects_points_in_bev_mask, 1)
        
        objects_in_bev = objects_points_num_in_bev>0
        objects_bev_points = objects_bev_points[objects_in_bev,:,:]

        for object_bev_points in objects_bev_points:
            object_bev_points[:, 0] = np.clip(object_bev_points[:,0],0,self.bev_width-1)
            object_bev_points[:, 1] = np.clip(object_bev_points[:,1],0,self.bev_height-1)

            cv2.fillConvexPoly(bb_array, object_bev_points, 255)
        return bb_array

    def crop_map(self,map):
        position = self.ego_transform.location
        center_x = int((-position.x* self.hdmap_px_per_meter + self.hdmap_half)*self.bev_hdmap_ratio)
        center_y = int((position.y*self.hdmap_px_per_meter + self.hdmap_half)*self.bev_hdmap_ratio)
        yaw = self.ego_transform.rotation.yaw
        map_crop = map[center_x-self.bev_diagonal:center_x+self.bev_diagonal, center_y-self.bev_diagonal:center_y+self.bev_diagonal]
        
        map_crop_rotate = imutils.rotate(map_crop,yaw)
        rot_center_x = map_crop_rotate.shape[0]//2
        rot_center_y = map_crop_rotate.shape[1]//2


        map_crop = map_crop_rotate[rot_center_x-(self.bev_height//2):rot_center_x+(self.bev_height//2), rot_center_y-(self.bev_width//2):rot_center_y+(self.bev_width//2)]
        map_crop = np.expand_dims(map_crop, -1)

        return map_crop





