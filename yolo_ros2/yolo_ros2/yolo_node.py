import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
import cv2
import numpy as np
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image
import tf2_ros
from visualization_msgs.msg import Marker, MarkerArray
from ultralytics import YOLO, solutions
from cv_bridge import CvBridge
from PIL import Image as Im
import pyrealsense2 as rs


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
        # Callback to handle depth image messages
        self.depth_image = self.br.imgmsg_to_cv2(msg, '32FC1')
        
        if self.depth_image is not None:
            # Log depth image dimensions
            height, width = self.depth_image.shape[:2]
            #self.get_logger().info(f'Depth image dimensions: {height}x{width}')

    def image_callback(self, msg):
        color_image = self.br.imgmsg_to_cv2(msg, 'bgr8')
        
        results = self.yolo_model.predict(color_image)
        print(results)
        boxes = results[0].boxes

        # Assuming a fixed depth for distance estimation
        fixed_depth = 1.0

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

            fx = 616.178588867188 
            fy = 616.587158203125
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
            self.get_logger().info(f'Distance between camera_link and object_{idx}: {distance} meters') # Attention, z values are from fixwd depthand not actual depth

            # Marker for visualization
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
            marker.scale.z = 0.1 
            marker.color.a = 0.7  
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker_array.markers.append(marker)
            # Publish the marker array
            self.marker_publisher.publish(marker_array)
        
        # Publisher for annotated images
        
        cv_image = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        results = self.yolo_model(cv_image)  # Perform the inference
        
        annotated_image = results[0]  # Get the annotated image from the first result
        im_array = annotated_image.plot()
        image_msg = Im.fromarray(im_array[..., ::-1])
        ##!!!!!Convert Pillow Image back to CV2 or Image_msg
        numpy_array = np.array(image_msg)
        image_msg = self.br.cv2_to_imgmsg(numpy_array, encoding='bgr8')

        # Publish the annotated image
        self.image_publisher.publish(image_msg)


            

def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

