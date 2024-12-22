import cv2, sys, glob, os
import numpy as np

os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

def main():

    cap = cv2.VideoCapture(0)
    
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.5) as pose:
    
      total = 0
      detected = 0
    
      key = -1
   
      print('Hit any key to terminate')

      freq = cv2.getTickFrequency()
      prev = cv2.getTickCount()
      frameNo = 0

      while key == -1:
    
        key = cv2.waitKey(10)
    
        ret, image = cap.read()
    
        if not ret:
          continue
    
        image_height, image_width, _ = image.shape
        # Convert the BGR image to RGB before processing.
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
        annotated_image = image.copy()
    
        if not results.pose_landmarks:
    
          cv2.imshow('annotated_image', annotated_image)
          continue
    
        # Draw pose landmarks on the image.
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    
        cv2.imshow('annotated_image', annotated_image)

        frameNo += 1
        curr = cv2.getTickCount()
        elapse = (curr - prev) / freq
        if elapse > 1:
            print('%.1f fps' % (frameNo / elapse))
            frameNo = 0
            prev= curr

if __name__ == '__main__':
    main()

