import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from geometry_msgs.msg import TransformStamped
import tf2_ros
from visualization_msgs.msg import Marker, MarkerArray
from ultralytics import YOLO

class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_node')
        # Subscriptions for image and depth topics
        self.image_subscription = self.create_subscription(Image, 'camera/camera/color/image_raw', self.image_callback, 10)
        self.depth_subscription = self.create_subscription(Image, 'camera/camera/depth/image_rect_raw', self.depth_callback, 10)
        # Bridge to convert ROS images to OpenCV format
        self.br = CvBridge()
        # Broadcaster for publishing transforms
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        # Publisher for visual markers
        self.marker_publisher = self.create_publisher(MarkerArray, '/yolo_markers', 10)
        
        # Load the YOLO model
        self.yolo_model = YOLO('/home/ninad/Downloads/best_landolts.pt')
        self.depth_image = None
        self.get_logger().info('YOLO Node has been started.')

    def depth_callback(self, msg):
        # Callback to handle depth image messages
        self.depth_image = self.br.imgmsg_to_cv2(msg, '32FC1')
        
        if self.depth_image is not None:
            # Log depth image dimensions
            height, width = self.depth_image.shape[:2]
            self.get_logger().info(f'Depth image dimensions: {height}x{width}')

    def image_callback(self, msg):
        color_image = self.br.imgmsg_to_cv2(msg, 'bgr8')
        
        results = self.yolo_model.predict(color_image)
        boxes = results[0].boxes  # Assuming single image

        # Assuming a fixed depth or scale factor for distance estimation
        fixed_depth = 1.0 # Example: Assuming all objects are approximately 2 meters away

        # Initialize marker_array to store markers
        marker_array = MarkerArray()

        for idx, box in enumerate(boxes):
            xmin, ymin, xmax, ymax = box.xyxy[0]
            x_center = int((xmin + xmax) / 2)
            y_center = int((ymin + ymax) / 2)

            if x_center < 0 or x_center >= color_image.shape[1] or y_center < 0 or y_center >= color_image.shape[0]:
                self.get_logger().warning(f'Invalid bounding box center coordinates ({x_center}, {y_center}). Skipping this detection.')
                continue

            # Use a fixed depth value for demonstration
            z = fixed_depth

            fx = 600.0  # Adjust according to your camera's parameters
            fy = 600.0  # Adjust according to your camera's parameters
            cx = color_image.shape[1] / 2
            cy = color_image.shape[0] / 2

            x = (x_center - cx) * z / fx
            y = (y_center - cy) * z / fy

            self.get_logger().info(f'Calculated coordinates: x={x}, y={y}, z={z}')

            # Broadcast the transform from camera_link to object_{idx}
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = 'camera_link'
            t.child_frame_id = f'object_{idx}'
            t.transform.translation.x = float(x)
            t.transform.translation.y = float(y)
            t.transform.translation.z = z
            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = 0.0
            t.transform.rotation.w = 1.0
            self.tf_broadcaster.sendTransform(t)

            # Calculate distance between camera_link and object_{idx}
            distance = np.sqrt(x**2 + y**2 + z**2)
            self.get_logger().info(f'Distance between camera_link and object_{idx}: {distance} meters')

            # Create a marker for visualization
            marker = Marker()
            marker.header.frame_id = 'camera_link'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'yolo_detections'
            marker.id = idx
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose.position.x = float(x)
            marker.pose.position.y = float(y)
            marker.pose.position.z = z
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = float((xmax - xmin) * z / fx)
            marker.scale.y = float((ymax - ymin) * z / fy)
            marker.scale.z = 0.1  # Arbitrary depth for the marker
            marker.color.a = 0.7  # Transparency
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker_array.markers.append(marker)

            # Define the desired TCP position relative to the object
            tcp_offset_z = 0.2  # Example offset, 0.5 meters in front of the object
            tcp_x = x
            tcp_y = y
            tcp_z = z + tcp_offset_z

            # Create a transform stamped message for the TCP
            tcp_transform = TransformStamped()
            tcp_transform.header.stamp = self.get_clock().now().to_msg()
            tcp_transform.header.frame_id = 'camera_link'
            tcp_transform.child_frame_id = 'tcp'
            tcp_transform.transform.translation.x = float(tcp_x)
            tcp_transform.transform.translation.y = float(tcp_y)
            tcp_transform.transform.translation.z = float(tcp_z)
            tcp_transform.transform.rotation.x = 0.0
            tcp_transform.transform.rotation.y = 0.0
            tcp_transform.transform.rotation.z = 0.0
            tcp_transform.transform.rotation.w = 1.0

            # Broadcast the transform for the TCP
            self.tf_broadcaster.sendTransform(tcp_transform)

        # Publish the marker array
        self.marker_publisher.publish(marker_array)





def main(args=None):
    # Main function to initialize and spin the ROS node
    rclpy.init(args=args)
    node = YoloNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
