import cv2
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from qreader import QReader

class QrCodeDetect(Node):
    def __init__(self):
        super().__init__("qrcode_node")

        self.create_subscription(Image, 'camera/camera/color/image_raw', self.camera_cb, 1)
        self.qreader = QReader(model_size='m')

    def camera_cb(self, msg):
        spot_gripper_img = CvBridge().imgmsg_to_cv2(msg, desired_encoding='bgr8')
        resized_img = cv2.resize(spot_gripper_img, (640, 480))
        grayscale = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        qreader_results = self.qreader.detect(image=grayscale)
        qreader_img = resized_img.copy()
        for result in qreader_results:
            if 'bbox_xyxy' in result:
                x1, y1, x2, y2 = result['bbox_xyxy']
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                 # Draw lines for the bounding box
                cv2.line(qreader_img, (x1, y1), (x2, y1), (0, 255, 0), 2)
                cv2.line(qreader_img, (x2, y1), (x2, y2), (0, 255, 0), 2)
                cv2.line(qreader_img, (x2, y2), (x1, y2), (0, 255, 0), 2)
                cv2.line(qreader_img, (x1, y2), (x1, y1), (0, 255, 0), 2)
            else:
                print("QReader result does not contain bounding box information.")

        width, height = spot_gripper_img.shape[1], spot_gripper_img.shape[0]
        resized_img = cv2.resize(qreader_img, (width // 2, height // 2))

        cv2.imshow("QR Code Detection", resized_img)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = QrCodeDetect()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
