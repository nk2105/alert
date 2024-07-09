import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
import cv2
import numpy as np
from geometry_msgs.msg import TransformStamped, Quaternion
from sensor_msgs.msg import Image
import tf2_ros
from visualization_msgs.msg import Marker, MarkerArray
from ultralytics import YOLO
from transforms3d.euler import euler2quat

class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_node')
        self.image_subscription = self.create_subscription(Image, 'camera/camera/color/image_raw', self.image_callback, 10)
        self.depth_subscription = self.create_subscription(Image, 'camera/camera/depth/image_rect_raw', self.depth_callback, 10)

        self.br = CvBridge()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.marker_publisher = self.create_publisher(MarkerArray, '/yolo_markers', 10)
        self.image_publisher = self.create_publisher(Image, '/annotations', 10)
        
        # Load the YOLO model
        self.yolo_model = YOLO('/home/ninad/Downloads/omni_points.pt')
        self.depth_image = None
        self.get_logger().info('YOLO Node has been started.')

    def depth_callback(self, msg):
        self.depth_image = self.br.imgmsg_to_cv2(msg, '32FC1')

    def image_callback(self, msg):
        #if self.depth_image is None:
        #    self.get_logger().warning('Depth image is not available yet.')
        #    return

        color_image = self.br.imgmsg_to_cv2(msg, 'bgr8')

        # Run YOLO inference
        results = self.yolo_model.predict(color_image)

        boxes = results[0].boxes
        keypoints = results[0].keypoints.data.cpu().numpy() if results[0].keypoints else []

        marker_array = MarkerArray()

        for idx, (box, keypoint) in enumerate(zip(boxes, keypoints)):
            xmin, ymin, xmax, ymax = box.xyxy[0]
            confidence = box.conf[0]
            class_id = box.cls[0]
            x_center = int((xmin + xmax) / 2)
            y_center = int((ymin + ymax) / 2)

            # Ensure depth image and depth value retrieval
        #    if y_center >= self.depth_image.shape[0] or x_center >= self.depth_image.shape[1]:
        #        self.get_logger().warning('Bounding box center out of depth image bounds.')
        #        continue
#
        #    z = float(self.depth_image[y_center, x_center])
        #    if np.isinf(z) or np.isnan(z) or z <= 0:
        #        self.get_logger().warning(f'Invalid depth value detected: {z}')
        #        continue

            z = 1.0 # Fixed depth assumed because depth sensor gives infinite distance
            fx = 616.178588867188 
            fy = 616.587158203125
            cx = color_image.shape[1] / 2
            cy = color_image.shape[0] / 2
            x = (x_center - cx) * z / fx
            y = (y_center - cy) * z / fy

            # Calculate orientation using keypoints
            orientation_quaternion = self.calculate_orientation_from_keypoints(keypoint)

            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = 'camera_link'
            t.child_frame_id = f'object_{int(class_id)}'
            t.transform.translation.x = float(x)
            t.transform.translation.y = float(y)
            t.transform.translation.z = z
            t.transform.rotation = orientation_quaternion
            self.tf_broadcaster.sendTransform(t)

            marker = Marker()
            marker.header.frame_id = 'camera_link'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'yolo_detections'
            marker.id = idx
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = float(x)
            marker.pose.position.y = float(y)
            marker.pose.position.z = z
            marker.pose.orientation = orientation_quaternion
            marker.scale.x = float((xmax - xmin) * z / fx)
            marker.scale.y = float((ymax - ymin) * z / fy)
            marker.scale.z = 0.1 
            marker.color.a = 0.7  
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker_array.markers.append(marker)

        self.marker_publisher.publish(marker_array)

        # Publisher for annotated images
        annotated_image = results[0].plot()
        image_msg = self.br.cv2_to_imgmsg(annotated_image, encoding='bgr8')
        self.image_publisher.publish(image_msg)


    def calculate_orientation_from_keypoints(self, keypoints):
        # Ensure keypoints are correctly accessed
        if len(keypoints) < 2:
            self.get_logger().error('Not enough keypoints to calculate orientation')
            return Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        
        try:
            front = keypoints[0][:2]
            back = keypoints[1][:2]
        except IndexError as e:
            self.get_logger().error(f'Error accessing keypoints: {e}')
            return Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

        delta_x = front[0] - back[0]
        delta_y = front[1] - back[1]
        angle = np.arctan2(delta_y, delta_x)

        quaternion = euler2quat(0, 0, angle)  # Assuming 2D orientation in the XY plane
        return Quaternion(x=quaternion[1], y=quaternion[2], z=quaternion[3], w=quaternion[0])

def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


