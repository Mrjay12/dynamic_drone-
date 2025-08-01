#!/usr/bin/env python3
import rospy
import numpy as np
import tf
from geometry_msgs.msg import PoseStamped, TransformStamped
from apriltag_ros.msg import AprilTagDetectionArray
from mavros_msgs.msg import AttitudeTarget
from tf.transformations import quaternion_from_euler, quaternion_multiply, euler_from_quaternion

class DownwardAprilTagYawNode:
    def __init__(self):
        rospy.init_node('downward_apriltag_yaw_node')
        
        # Subscribers
        self.tag_sub = rospy.Subscriber('/tag_detections', AprilTagDetectionArray, self.tag_callback)
        
        # Publishers
        self.attitude_pub = rospy.Publisher('/mavros/setpoint_raw/attitude', AttitudeTarget, queue_size=10)
        self.vision_pub = rospy.Publisher('/mavros/vision_pose/pose', PoseStamped, queue_size=10)
        
        # Parameters
        self.tag_id = 0          # Target AprilTag ID
        self.camera_frame = "oak_stereo_left_frame"
        self.body_frame = "base_link"
        
        # TF broadcaster for camera-to-body transform
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.timer = rospy.Timer(rospy.Duration(0.05), self.publish_camera_transform)

    def publish_camera_transform(self, event):
        # Camera is mounted downward: -90° rotation around X-axis
        q = quaternion_from_euler(-np.pi/2, 0, 0)  # -90° around X
        self.tf_broadcaster.sendTransform(
            (0, 0, 0), q, rospy.Time.now(), 
            self.camera_frame, self.body_frame
        )

    def tag_callback(self, msg):
        if not msg.detections:
            return
            
        for detection in msg.detections:
            if detection.id[0] == self.tag_id:
                # Get tag pose in camera frame
                tag_pose_cam = detection.pose.pose
                
                # Transform to body frame
                tag_pose_body = self.transform_to_body_frame(tag_pose_cam)
                
                # Extract yaw from tag's orientation
                yaw = self.extract_yaw_from_tag(tag_pose_body.orientation)
                
                # Send yaw to CUAV V5+
                self.send_yaw_to_fc(yaw)
                
                # Send position to CUAV V5+ (for platform-relative positioning)
                self.send_position_to_fc(tag_pose_body.position)

    def transform_to_body_frame(self, cam_pose):
        # Apply camera-to-body transformation (-90° around X)
        q_cam = [cam_pose.orientation.x, cam_pose.orientation.y, 
                 cam_pose.orientation.z, cam_pose.orientation.w]
        q_rot = quaternion_from_euler(-np.pi/2, 0, 0)  # Camera rotation
        q_body = quaternion_multiply(q_rot, q_cam)  # Rotate camera orientation
        
        # Transform position (Z becomes Y in body frame due to rotation)
        pos = cam_pose.position
        body_pos = PoseStamped()
        body_pos.pose.position.x = pos.x
        body_pos.pose.position.y = -pos.z  # Z (down) becomes Y (right)
        body_pos.pose.position.z = pos.y   # Y (forward) becomes Z (up)
        body_pos.pose.orientation.x = q_body[0]
        body_pos.pose.orientation.y = q_body[1]
        body_pos.pose.orientation.z = q_body[2]
        body_pos.pose.orientation.w = q_body[3]
        
        return body_pos.pose

    def extract_yaw_from_tag(self, orientation):
        # Convert quaternion to Euler angles
        q = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(q)
        return yaw

    def send_yaw_to_fc(self, yaw):
        msg = AttitudeTarget()
        msg.header.stamp = rospy.Time.now()
        msg.type_mask = 7  # Ignore roll/pitch/thrust
        q = quaternion_from_euler(0, 0, yaw)
        msg.orientation = q
        self.attitude_pub.publish(msg)

    def send_position_to_fc(self, position):
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        msg.pose.position = position
        self.vision_pub.publish(msg)

if __name__ == '__main__':
    try:
        DownwardAprilTagYawNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
