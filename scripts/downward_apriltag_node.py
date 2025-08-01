#!/usr/bin/env python3
import rospy
import numpy as np
import tf
from geometry_msgs.msg import PoseStamped, TwistStamped
from apriltag_ros.msg import AprilTagDetectionArray
from mavros_msgs.msg import AttitudeTarget, LandingTarget, State
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Empty
from tf.transformations import quaternion_from_euler, quaternion_multiply, euler_from_quaternion

class UnifiedFlightController:
    def __init__(self):
        rospy.init_node('unified_flight_controller')
        
        # Subscribers
        self.tag_sub = rospy.Subscriber('/tag_detections', AprilTagDetectionArray, self.tag_callback)
        self.vio_sub = rospy.Subscriber('/stereo_inertial_node/odom', Odometry, self.vio_callback)
        self.state_sub = rospy.Subscriber('/mavros/state', State, self.state_callback)
        self.takeoff_cmd_sub = rospy.Subscriber('/takeoff/command', Bool, self.takeoff_command_callback)
        self.landing_cmd_sub = rospy.Subscriber('/landing/command', Bool, self.landing_command_callback)
        
        # Publishers
        self.attitude_pub = rospy.Publisher('/mavros/setpoint_raw/attitude', AttitudeTarget, queue_size=10)
        self.vision_pub = rospy.Publisher('/mavros/vision_pose/pose', PoseStamped, queue_size=10)
        self.speed_pub = rospy.Publisher('/mavros/vision_speed/speed', TwistStamped, queue_size=10)
        self.landing_target_pub = rospy.Publisher('/mavros/landing_target', LandingTarget, queue_size=10)
        
        # Parameters
        self.tag_id = 0
        self.TAG_SIZE = 0.1
        self.FOCAL_LENGTH = 800
        self.TAKEOFF_ALTITUDE = 15.0
        self.LANDING_FINAL_ALTITUDE = 0.3
        self.LANDING_DESCENT_ALTITUDE = 2.0
        
        # Fusion weights
        self.ALPHA_POS = 0.1
        self.ALPHA_YAW = 0.1
        self.ALPHA_VEL = 0.8
        
        # State machine
        self.STATE_IDLE = 0
        self.STATE_TAKEOFF_INITIATED = 1
        self.STATE_TAKEOFF_SEARCHING = 2
        self.STATE_TAKEOFF_ASCENT = 3
        self.STATE_NORMAL_FLIGHT = 4
        self.STATE_LANDING_INITIATED = 5
        self.STATE_LANDING_SEARCHING = 6
        self.STATE_LANDING_DESCENT = 7
        self.STATE_LANDING_FINAL = 8
        self.STATE_LANDED = 9
        
        self.flight_state = self.STATE_IDLE
        self.armed = False
        
        # State variables
        self.vio_data = None
        self.tag_data = None
        self.fused_pose = None
        self.fused_twist = None
        self.last_tag_time = rospy.Time(0)
        self.tag_valid_duration = rospy.Duration(0.5)
        
        # Initialize with VIO if available
        rospy.Timer(rospy.Duration(1.0), self.check_initialization)
        
        # TF broadcaster
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.timer = rospy.Timer(rospy.Duration(0.05), self.publish_camera_transform)

    def publish_camera_transform(self, event):
        q = quaternion_from_euler(-np.pi/2, 0, 0)
        self.tf_broadcaster.sendTransform(
            (0, 0, 0), q, rospy.Time.now(), 
            "oak_stereo_left_frame", "base_link"
        )

    def check_initialization(self, event):
        if self.vio_data is None:
            rospy.logwarn_throttle(5.0, "Waiting for VIO data...")
        else:
            rospy.loginfo("VIO data received, system initialized")

    def state_callback(self, msg):
        self.armed = msg.armed

    def takeoff_command_callback(self, msg):
        if msg.data and self.flight_state == self.STATE_IDLE:
            self.flight_state = self.STATE_TAKEOFF_INITIATED
            rospy.loginfo("Takeoff sequence initiated")

    def landing_command_callback(self, msg):
        if msg.data and self.flight_state == self.STATE_NORMAL_FLIGHT:
            self.flight_state = self.STATE_LANDING_INITIATED
            rospy.loginfo("Landing sequence initiated")

    def vio_callback(self, msg):
        self.vio_data = msg
        vio_pose = msg.pose.pose
        vio_twist = msg.twist.twist
        
        if self.fused_pose is None:
            self.fused_pose = vio_pose
            self.fused_twist = vio_twist
            return
        
        # Update fused velocity
        self.fused_twist.linear.x = self.ALPHA_VEL * vio_twist.linear.x + (1 - self.ALPHA_VEL) * self.fused_twist.linear.x
        self.fused_twist.linear.y = self.ALPHA_VEL * vio_twist.linear.y + (1 - self.ALPHA_VEL) * self.fused_twist.linear.y
        self.fused_twist.linear.z = self.ALPHA_VEL * vio_twist.linear.z + (1 - self.ALPHA_VEL) * self.fused_twist.linear.z
        
        # Update fused position if no recent tag
        if (rospy.Time.now() - self.last_tag_time) > self.tag_valid_duration:
            self.fused_pose.position.x = vio_pose.position.x
            self.fused_pose.position.y = vio_pose.position.y
            self.fused_pose.position.z = vio_pose.position.z
        
        # Update state machine
        self.update_flight_state()
        
        # Always publish fused data
        self.publish_fused_data()

    def tag_callback(self, msg):
        current_time = rospy.Time.now()
        
        for detection in msg.detections:
            if detection.id[0] == self.tag_id:
                tag_pose_cam = detection.pose.pose
                tag_pose_body = self.transform_to_body_frame(tag_pose_cam)
                
                self.tag_data = tag_pose_body
                self.last_tag_time = current_time
                
                self.fuse_apriltag_data(tag_pose_body)
                self.process_tag_for_flight(tag_pose_body, detection)
                break

    def update_flight_state(self):
        current_altitude = self.vio_data.pose.pose.position.z if self.vio_data else 0.0
        
        # Takeoff state transitions
        if self.flight_state == self.STATE_TAKEOFF_INITIATED:
            if self.armed:
                self.flight_state = self.STATE_TAKEOFF_SEARCHING
                rospy.loginfo("Searching for takeoff tag")
        
        elif self.flight_state == self.STATE_TAKEOFF_SEARCHING:
            if self.tag_data is not None:
                self.flight_state = self.STATE_TAKEOFF_ASCENT
                rospy.loginfo("Takeoff tag detected, ascending")
        
        elif self.flight_state == self.STATE_TAKEOFF_ASCENT:
            if current_altitude >= self.TAKEOFF_ALTITUDE:
                self.flight_state = self.STATE_NORMAL_FLIGHT
                rospy.loginfo("Reached takeoff altitude, normal flight")
        
        # Landing state transitions
        elif self.flight_state == self.STATE_LANDING_INITIATED:
            self.flight_state = self.STATE_LANDING_SEARCHING
            rospy.loginfo("Searching for landing tag")
        
        elif self.flight_state == self.STATE_LANDING_SEARCHING:
            if self.tag_data is not None:
                self.flight_state = self.STATE_LANDING_DESCENT
                rospy.loginfo("Landing tag detected, descending")
        
        elif self.flight_state == self.STATE_LANDING_DESCENT:
            if current_altitude <= self.LANDING_DESCENT_ALTITUDE:
                self.flight_state = self.STATE_LANDING_FINAL
                rospy.loginfo("Final approach phase")
        
        elif self.flight_state == self.STATE_LANDING_FINAL:
            if current_altitude <= self.LANDING_FINAL_ALTITUDE:
                self.flight_state = self.STATE_LANDED
                rospy.loginfo("Landing complete")
                self.disarm_motors()

    def process_tag_for_flight(self, tag_pose, detection):
        if self.flight_state == self.STATE_TAKEOFF_ASCENT:
            self.process_takeoff_tag(tag_pose)
        elif self.flight_state in [self.STATE_LANDING_DESCENT, self.STATE_LANDING_FINAL]:
            self.process_landing_tag(tag_pose, detection)

    def process_takeoff_tag(self, tag_pose):
        offset_x = tag_pose.position.x
        offset_y = tag_pose.position.y
        
        max_vel = 0.5
        vel_x = np.clip(-offset_x * 0.5, -max_vel, max_vel)
        vel_y = np.clip(-offset_y * 0.5, -max_vel, max_vel)
        
        twist_msg = TwistStamped()
        twist_msg.header.stamp = rospy.Time.now()
        twist_msg.twist.linear.x = vel_x
        twist_msg.twist.linear.y = vel_y
        twist_msg.twist.linear.z = 1.0
        self.speed_pub.publish(twist_msg)

    def process_landing_tag(self, tag_pose, detection):
        offset_x = tag_pose.position.x
        offset_y = tag_pose.position.y
        
        tag_size_pixels = detection.size[0]
        altitude = (self.TAG_SIZE * self.FOCAL_LENGTH) / tag_size_pixels if tag_size_pixels > 0 else 2.0
        
        self.publish_landing_target(offset_x, offset_y, altitude)
        
        max_vel = 0.3
        vel_x = np.clip(-offset_x * 0.5, -max_vel, max_vel)
        vel_y = np.clip(-offset_y * 0.5, -max_vel, max_vel)
        
        descent_rate = -0.5 if self.flight_state == self.STATE_LANDING_DESCENT else -0.2
        
        twist_msg = TwistStamped()
        twist_msg.header.stamp = rospy.Time.now()
        twist_msg.twist.linear.x = vel_x
        twist_msg.twist.linear.y = vel_y
        twist_msg.twist.linear.z = descent_rate
        self.speed_pub.publish(twist_msg)

    def publish_landing_target(self, offset_x, offset_y, altitude):
        msg = LandingTarget()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "landing_target"
        msg.target_num = 0
        msg.frame = 8
        msg.angle_x = np.arctan2(offset_x, altitude)
        msg.angle_y = np.arctan2(offset_y, altitude)
        msg.distance = altitude
        msg.size_x = self.TAG_SIZE
        msg.size_y = self.TAG_SIZE
        self.landing_target_pub.publish(msg)

    def disarm_motors(self):
        rospy.loginfo("Disarming motors")
        # Implement MAVLink disarm command here

    def fuse_apriltag_data(self, tag_pose):
        if self.fused_pose is None:
            self.fused_pose = tag_pose
            return
        
        self.fused_pose.position.x = self.ALPHA_POS * tag_pose.position.x + (1 - self.ALPHA_POS) * self.fused_pose.position.x
        self.fused_pose.position.y = self.ALPHA_POS * tag_pose.position.y + (1 - self.ALPHA_POS) * self.fused_pose.position.y
        self.fused_pose.position.z = self.ALPHA_POS * tag_pose.position.z + (1 - self.ALPHA_POS) * self.fused_pose.position.z
        
        tag_yaw = self.extract_yaw_from_tag(tag_pose.orientation)
        fused_yaw = self.ALPHA_YAW * tag_yaw + (1 - self.ALPHA_YAW) * self.extract_yaw_from_tag(self.fused_pose.orientation)
        
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

    def publish_fused_data(self):
        if self.fused_pose is None or self.fused_twist is None:
            return
        
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "map"
        pose_msg.pose = self.fused_pose
        self.vision_pub.publish(pose_msg)
        
        twist_msg = TwistStamped()
        twist_msg.header.stamp = rospy.Time.now()
        twist_msg.header.frame_id = "map"
        twist_msg.twist = self.fused_twist
        self.speed_pub.publish(twist_msg)
        
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
        UnifiedFlightController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
