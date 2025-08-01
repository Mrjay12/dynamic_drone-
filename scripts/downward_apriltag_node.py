#!/usr/bin/env python3
import rospy
import numpy as np
import tf
from geometry_msgs.msg import PoseStamped, TwistStamped, PoseWithCovarianceStamped
from apriltag_ros.msg import AprilTagDetectionArray
from mavros_msgs.msg import AttitudeTarget
from nav_msgs.msg import Odometry
from tf.transformations import quaternion_from_euler, quaternion_multiply, euler_from_quaternion

class FusedVisionNode:
    def __init__(self):
        rospy.init_node('fused_vision_node')
        
        # Subscribers
        self.tag_sub = rospy.Subscriber('/tag_detections', AprilTagDetectionArray, self.tag_callback)
        self.vio_sub = rospy.Subscriber('/stereo_inertial_node/odom', Odometry, self.vio_callback)
        
        # Publishers
        self.attitude_pub = rospy.Publisher('/mavros/setpoint_raw/attitude', AttitudeTarget, queue_size=10)
        self.vision_pub = rospy.Publisher('/mavros/vision_pose/pose', PoseStamped, queue_size=10)
        self.speed_pub = rospy.Publisher('/mavros/vision_speed/speed', TwistStamped, queue_size=10)
        
        # Parameters
        self.tag_id = 0
        self.camera_frame = "oak_stereo_left_frame"
        self.body_frame = "base_link"
        
        # Fusion weights
        self.ALPHA_POS = 0.1  # AprilTag position weight
        self.ALPHA_YAW = 0.1  # AprilTag yaw weight
        self.ALPHA_VEL = 0.8  # VIO velocity weight
        
        # State variables
        self.vio_data = None
        self.tag_data = None
        self.fused_pose = None
        self.fused_twist = None
        self.last_tag_time = rospy.Time(0)
        self.tag_valid_duration = rospy.Duration(0.5)  # Tag data valid for 0.5s
        
        # Initialize with VIO if available
        rospy.Timer(rospy.Duration(1.0), self.check_initialization)
        
        # TF broadcaster
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.timer = rospy.Timer(rospy.Duration(0.05), self.publish_camera_transform)

    def publish_camera_transform(self, event):
        q = quaternion_from_euler(-np.pi/2, 0, 0)
        self.tf_broadcaster.sendTransform(
            (0, 0, 0), q, rospy.Time.now(), 
            self.camera_frame, self.body_frame
        )

    def check_initialization(self, event):
        if self.vio_data is None:
            rospy.logwarn_throttle(5.0, "Waiting for VIO data...")
        else:
            rospy.loginfo("VIO data received, system initialized")

    def vio_callback(self, msg):
        """Process VIO data"""
        self.vio_data = msg
        
        # Extract VIO components
        vio_pose = msg.pose.pose
        vio_twist = msg.twist.twist
        
        # Initialize fused data if needed
        if self.fused_pose is None:
            self.fused_pose = vio_pose
            self.fused_twist = vio_twist
            return
        
        # Update fused velocity (always trust VIO more)
        self.fused_twist.linear.x = self.ALPHA_VEL * vio_twist.linear.x + (1 - self.ALPHA_VEL) * self.fused_twist.linear.x
        self.fused_twist.linear.y = self.ALPHA_VEL * vio_twist.linear.y + (1 - self.ALPHA_VEL) * self.fused_twist.linear.y
        self.fused_twist.linear.z = self.ALPHA_VEL * vio_twist.linear.z + (1 - self.ALPHA_VEL) * self.fused_twist.linear.z
        
        # Update fused position (only if no recent AprilTag)
        if (rospy.Time.now() - self.last_tag_time) > self.tag_valid_duration:
            self.fused_pose.position.x = vio_pose.position.x
            self.fused_pose.position.y = vio_pose.position.y
            self.fused_pose.position.z = vio_pose.position.z
        
        # Always publish fused data
        self.publish_fused_data()

    def tag_callback(self, msg):
        """Process AprilTag data"""
        current_time = rospy.Time.now()
        
        for detection in msg.detections:
            if detection.id[0] == self.tag_id:
                # Transform tag to body frame
                tag_pose_cam = detection.pose.pose
                tag_pose_body = self.transform_to_body_frame(tag_pose_cam)
                
                # Store tag data
                self.tag_data = tag_pose_body
                self.last_tag_time = current_time
                
                # Fuse with VIO
                self.fuse_apriltag_data(tag_pose_body)
                break

    def fuse_apriltag_data(self, tag_pose):
        """Fuse AprilTag data with VIO"""
        if self.fused_pose is None:
            self.fused_pose = tag_pose
            return
        
        # Fuse position (trust AprilTag more)
        self.fused_pose.position.x = self.ALPHA_POS * tag_pose.position.x + (1 - self.ALPHA_POS) * self.fused_pose.position.x
        self.fused_pose.position.y = self.ALPHA_POS * tag_pose.position.y + (1 - self.ALPHA_POS) * self.fused_pose.position.y
        self.fused_pose.position.z = self.ALPHA_POS * tag_pose.position.z + (1 - self.ALPHA_POS) * self.fused_pose.position.z
        
        # Fuse yaw (trust AprilTag more)
        tag_yaw = self.extract_yaw_from_tag(tag_pose.orientation)
        fused_yaw = self.ALPHA_YAW * tag_yaw + (1 - self.ALPHA_YAW) * self.extract_yaw_from_tag(self.fused_pose.orientation)
        
        # Update fused orientation
        roll, pitch, _ = euler_from_quaternion([
            self.fused_pose.orientation.x,
            self.fused_pose.orientation.y,
            self.fused_pose.orientation.z,
            self.fused_pose.orientation.w
        ])
        q = quaternion_from_euler(roll, pitch, fused_yaw)
        self.fused_pose.orientation.x = q[0]
        self.fused_pose.orientation.y = q[1]
        self.fused_pose.orientation.z = q[2]
        self.fused_pose.orientation.w = q[3]
        
        # Publish fused data
        self.publish_fused_data()

    def publish_fused_data(self):
        """Publish fused pose and velocity"""
        if self.fused_pose is None or self.fused_twist is None:
            return
        
        # Publish fused pose
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "map"
        pose_msg.pose = self.fused_pose
        self.vision_pub.publish(pose_msg)
        
        # Publish fused velocity
        twist_msg = TwistStamped()
        twist_msg.header.stamp = rospy.Time.now()
        twist_msg.header.frame_id = "map"
        twist_msg.twist = self.fused_twist
        self.speed_pub.publish(twist_msg)
        
        # Extract and send yaw
        yaw = self.extract_yaw_from_tag(self.fused_pose.orientation)
        self.send_yaw_to_fc(yaw)

    def transform_to_body_frame(self, cam_pose):
        q_cam = [cam_pose.orientation.x, cam_pose.orientation.y, 
                 cam_pose.orientation.z, cam_pose.orientation.w]
        q_rot = quaternion_from_euler(-np.pi/2, 0, 0)
        q_body = quaternion_multiply(q_rot, q_cam)
        
        pos = cam_pose.position
        body_pos = PoseStamped()
        body_pos.pose.position.x = pos.x
        body_pos.pose.position.y = -pos.z
        body_pos.pose.position.z = pos.y
        body_pos.pose.orientation.x = q_body[0]
        body_pos.pose.orientation.y = q_body[1]
        body_pos.pose.orientation.z = q_body[2]
        body_pos.pose.orientation.w = q_body[3]
        
        return body_pos.pose

    def extract_yaw_from_tag(self, orientation):
        q = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(q)
        return yaw

    def send_yaw_to_fc(self, yaw):
        msg = AttitudeTarget()
        msg.header.stamp = rospy.Time.now()
        msg.type_mask = 7
        q = quaternion_from_euler(0, 0, yaw)
        msg.orientation = q
        self.attitude_pub.publish(msg)

if __name__ == '__main__':
    try:
        FusedVisionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
