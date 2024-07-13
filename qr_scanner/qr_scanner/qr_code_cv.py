import cv2
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class QrCodeDetect(Node):
    def __init__(self):
        super().__init__("qrcode_node")

        self.create_subscription(Image, 'camera/camera/color/image_raw', self.camera_cb, 1)
        self.qreader = cv2.QRCodeDetector()
        self.timer = self.create_timer(1.0, self.timer_cb)

    def camera_cb(self, msg):
        spot_gripper_img = CvBridge().imgmsg_to_cv2(msg, desired_encoding='bgr8')
        resized_img = cv2.resize(spot_gripper_img, (640, 480))
        grayscale = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        qr_code_img = resized_img.copy()
        self.val, data, bbox = self.qreader.detectAndDecode(grayscale)

        if data is not None and data.size > 0:
            if bbox is not None and len(bbox) > 0:
                points = bbox[0]
                if len(points) == 4:
                    # Extract the points from the bounding box
                    ptA, ptB, ptC, ptD = points

                    # Convert points to integer tuples
                    ptA = (int(ptA[0]), int(ptA[1]))
                    ptB = (int(ptB[0]), int(ptB[1]))
                    ptC = (int(ptC[0]), int(ptC[1]))
                    ptD = (int(ptD[0]), int(ptD[1]))

                    # Draw the bounding box on the grayscale image
                    cv2.line(qr_code_img, ptA, ptB, (0, 255, 0), 2)
                    cv2.line(qr_code_img, ptB, ptC, (0, 255, 0), 2)
                    cv2.line(qr_code_img, ptC, ptD, (0, 255, 0), 2)
                    cv2.line(qr_code_img, ptD, ptA, (0, 255, 0), 2)

        width, height = spot_gripper_img.shape[1], spot_gripper_img.shape[0]
        resized_img = cv2.resize(qr_code_img, (width // 2, height // 2))

        cv2.imshow("QR Code Detection", resized_img)
        cv2.waitKey(1)

    def timer_cb(self):
        if self.val:
            self.get_logger().info(self.val)


def main(args=None):
    rclpy.init(args=args)
    node = QrCodeDetect()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
