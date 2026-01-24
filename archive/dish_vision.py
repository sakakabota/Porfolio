# dish_vision.py
# Computer vision component for Dish-n-Dash robotic dish sorting system

import rospy
import torch
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from std_msgs.msg import String
import pyrealsense2 as rs
from transformers import DetrImageProcessor, DetrForObjectDetection
import message_filters
import tf2_ros
import tf2_geometry_msgs
from tf.transformations import euler_from_quaternion, quaternion_from_euler

# Constants
DETECTION_THRESHOLD = 0.7
FOOD_LABELS = [53, 55, 52, 54]  # Labels for apple, banana, orange, sandwich
UTENSIL_LABELS = [84, 48, 89]   # Labels for spoon, fork, cup
LABEL_MAP = {
    53: 'apple',
    55: 'banana',
    52: 'orange',
    54: 'sandwich',
    84: 'spoon',
    48: 'fork',
    89: 'cup',
    56: 'bowl'
}

class DishVisionSystem:
    def __init__(self):
        rospy.init_node('dish_vision_node')
        
        # Load DETR model and processor
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        self.model.eval()
        
        # RealSense camera setup
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)
        
        # ROS bridge and publishers
        self.bridge = CvBridge()
        self.object_pub = rospy.Publisher('/detected_objects', String, queue_size=10)
        self.image_pub = rospy.Publisher('/detection_visualization', Image, queue_size=10)
        
        # TF buffer for coordinate transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Initialize tracking variables
        self.bowl_position = None
        self.detected_items = []
        
        rospy.loginfo("Dish Vision System initialized successfully")
    
    def preprocess_for_detr(self, frame):
        """Preprocess the frame for DETR model input"""
        inputs = self.processor(images=frame, return_tensors="pt")
        return inputs
    
    def camera_to_robot_transform(self, box, depth_frame):
        """Transform camera coordinates to robot base coordinates"""
        # Extract bounding box center
        x_center = int((box[0] + box[2]) / 2)
        y_center = int((box[1] + box[3]) / 2)
        
        # Get depth at center point (in meters)
        depth = depth_frame.get_distance(x_center, y_center)
        
        # Deproject to 3D point in camera frame
        depth_intrinsics = rs.video_stream_profile(
            self.profile.get_stream(rs.stream.depth)).get_intrinsics()
        camera_point = rs.rs2_deproject_pixel_to_point(
            depth_intrinsics, [x_center, y_center], depth)
        
        # Transform to robot base frame
        try:
            trans = self.tf_buffer.lookup_transform(
                'base', 'camera_color_optical_frame', rospy.Time())
            
            # Create point in camera frame
            point_camera = tf2_geometry_msgs.PointStamped()
            point_camera.header.frame_id = "camera_color_optical_frame"
            point_camera.header.stamp = rospy.Time.now()
            point_camera.point.x = camera_point[0]
            point_camera.point.y = camera_point[1]
            point_camera.point.z = camera_point[2]
            
            # Transform to robot frame
            point_robot = tf2_ros.tf2_geometry_msgs.do_transform_point(
                point_camera, trans)
            
            return (point_robot.point.x, point_robot.point.y, point_robot.point.z)
        
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, 
                tf2_ros.ExtrapolationException) as e:
            rospy.logerr("TF Error: %s", e)
            return None
    
    def process_frame(self):
        """Process current camera frames for object detection"""
        try:
            # Wait for frames
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            
            # Get aligned color and depth frames
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                return
            
            # Convert to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            
            # Preprocess for DETR
            inputs = self.preprocess_for_detr(color_image)
            
            # Run detection
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Post-process DETR output
            results = self.processor.post_process_object_detection(
                outputs, threshold=DETECTION_THRESHOLD)[0]
            
            detected_objects = []
            visualization_img = color_image.copy()
            
            # Process each detection
            for score, label, box in zip(
                    results["scores"], results["labels"], results["boxes"]):
                
                # Convert box coordinates to integers
                box = [int(i) for i in box.tolist()]
                
                # Get label and category
                label_id = label.item()
                if label_id not in LABEL_MAP:
                    continue
                    
                label_name = LABEL_MAP[label_id]
                
                # Determine object category
                if label_id == 56:  # Bowl
                    category = 'bowl'
                    color = (255, 0, 255)  # Magenta for bowl
                    
                    # Store bowl position for later use
                    self.bowl_position = self.camera_to_robot_transform(box, depth_frame)
                    
                elif label_id in FOOD_LABELS:
                    category = 'food'
                    color = (0, 255, 0)  # Green for food
                elif label_id in UTENSIL_LABELS:
                    category = 'utensil'
                    color = (0, 0, 255)  # Blue for utensils
                else:
                    continue
                
                # Get 3D position in robot frame
                position = self.camera_to_robot_transform(box, depth_frame)
                if not position:
                    continue
                
                # Create object data
                object_data = {
                    'label': label_name,
                    'category': category,
                    'position': position,
                    'confidence': score.item(),
                    'box': box
                }
                detected_objects.append(object_data)
                
                # Draw bounding box for visualization
                cv2.rectangle(visualization_img, 
                             (box[0], box[1]), 
                             (box[2], box[3]), 
                             color, 2)
                
                # Add label text
                cv2.putText(visualization_img, 
                           f"{label_name}: {score:.2f}", 
                           (box[0], box[1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Publish results
            if detected_objects:
                self.detected_items = detected_objects
                self.object_pub.publish(str(detected_objects))
            
            # Publish visualization
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(visualization_img, "bgr8"))
            
        except Exception as e:
            rospy.logerr("Error in vision processing: %s", str(e))
    
    def get_bowl_position(self):
        """Return the latest detected bowl position"""
        return self.bowl_position
    
    def get_detected_items(self):
        """Return the list of detected items"""
        return self.detected_items
    
    def run(self):
        """Main processing loop"""
        rate = rospy.Rate(10)  # 10 Hz
        
        while not rospy.is_shutdown():
            self.process_frame()
            rate.sleep()
            
    def shutdown(self):
        """Clean shutdown"""
        self.pipeline.stop()
        rospy.loginfo("Dish Vision System shutting down")

if __name__ == '__main__':
    try:
        vision_system = DishVisionSystem()
        vision_system.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        if 'vision_system' in locals():
            vision_system.shutdown()
