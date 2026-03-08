#!/usr/bin/env python

import rospy
import cv2
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

class VideoRecorderNode:
    def __init__(self):
        rospy.init_node('video_recorder_node', anonymous=True)
        
        self.bridge = CvBridge()
        self.crop_width = 720   # Desired width of the rectangle
        self.crop_height = 720
        self.is_recording = False
        self.video_writer = None
        self.video_path = ""
        
        # Parameters
        self.fps = 15.0  # Adjust based on your camera's actual output
        
        # Subscribers
        # Listen for the trigger (file path to start, "0" to stop)
        rospy.Subscriber("/activate_record_video", String, self.trigger_callback)
        # Listen for the raw camera feed
        rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        
        rospy.loginfo("Video Recorder Node Initialized. Waiting for trigger...")

    def trigger_callback(self, msg):
        command = msg.data.strip()
        
        if command == "0":
            if self.is_recording:
                self.stop_recording()
            else:
                rospy.logwarn("Received stop signal, but not currently recording.")
        else:
            if not self.is_recording:
                self.video_path = command
                self.start_recording()
            else:
                rospy.logwarn(f"Already recording to {self.video_path}. Stop current record first.")

    def start_recording(self):
        self.is_recording = True
        rospy.loginfo(f"Started recording to: {self.video_path}")

    def stop_recording(self):
        self.is_recording = False
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        rospy.loginfo("Recording stopped and file saved.")

    def image_callback(self, data):
        if not self.is_recording:
            return

        try:
            # 1. Convert ROS Image to NumPy (RGB)
            full_frame = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
            
            # 2. Calculate Center Crop Coordinates
            H, W, _ = full_frame.shape
            
            # Ensure crop size isn't larger than the actual frame
            cw = min(self.crop_width, W)
            ch = min(self.crop_height, H)
            
            start_x = (W - cw) // 2
            start_y = (H - ch) // 2
            
            # 3. Perform the Crop
            cropped_frame = full_frame[start_y:start_y+ch, start_x:start_x+cw]
            
            # 4. Initialize VideoWriter (Must use cropped dimensions)
            if self.video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                # OpenCV expects (Width, Height)
                self.video_writer = cv2.VideoWriter(self.video_path, fourcc, self.fps, (cw, ch))
            
            # 5. Convert to BGR for OpenCV and Write
            # bgr_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_RGB2BGR)
            self.video_writer.write(cropped_frame)
        except Exception as e:
            rospy.logerr(f"Error in image processing: {e}")

if __name__ == '__main__':
    try:
        recorder = VideoRecorderNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass