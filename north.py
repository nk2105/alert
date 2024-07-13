import rclpy
from rclpy.node import Node
from sensor_msgs.msg import MagneticField
from visualization_msgs.msg import Marker
import math

class MagnetometerNode(Node):
    def __init__(self):
        super().__init__('magnetometer_node')

        # Declination data for Aachen, Germany
        self.declination_degrees = 2 + 42/60  # Convert degrees and minutes to decimal degrees
        declination_uncertainty = 23/60  # Convert uncertainty to decimal degrees
        annual_change = 11/60  # Convert annual change to decimal degrees

        self.subscription = self.create_subscription(
            MagneticField,
            '/olive/imu/id01/magnetometer',
            self.magnetometer_callback,
            10)

        self.marker_publisher_ = self.create_publisher(Marker, 'visualization_marker', 10)
        self.callback_timer = self.create_timer(2.0, self.timer_callback)

        self.heading = 0.0  # Initialize heading
        self.heading_deg = 0.0  # Initialize heading_deg

    def magnetometer_callback(self, msg):
        # Calculate heading from magnetometer x and y data
        self.heading = math.atan2(msg.magnetic_field.y, msg.magnetic_field.x)

        # Convert heading to degrees
        self.heading_deg = self.heading * (180.0 / math.pi)

        # Apply declination correction
        self.heading_deg += self.declination_degrees

        # Ensure heading is 0-360
        self.heading_deg = self.heading_deg % 360

    def timer_callback(self):
        self.get_logger().info('Heading: {:.2f} deg'.format(self.heading_deg))
        self.publish_marker()

    def publish_marker(self):
        marker = Marker()
        marker.header.frame_id = "olive"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "heading"
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = 1.0
        marker.pose.position.y = 1.0
        marker.pose.position.z = 1.0
        marker.scale.z = 0.5  # Height of the text
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0  # Alpha
        marker.text = 'Heading: {:.2f} deg'.format(self.heading_deg)
        self.marker_publisher_.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    magnetometer_node = MagnetometerNode()
    try:
        rclpy.spin(magnetometer_node)
    except KeyboardInterrupt:
        pass  # allow Ctrl-C to end spin()

    magnetometer_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
