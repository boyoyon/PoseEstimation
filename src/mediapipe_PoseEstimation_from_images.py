import cv2, sys, glob, os
import numpy as np

os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

def main():

    argv = sys.argv
    argc = len(argv)
    
    if argc < 2:
        print('%s estimates pose' % argv[0])
        print('%s <wildcard for images>' % argv[0])
        quit()
    
    paths = glob.glob(argv[1])
    
    nrImages = len(paths)
    
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.5) as pose:
    
        for idx, path in enumerate(paths):
        
            print('%d/%d: ' % (idx + 1, nrImages), end='')
    
            image = cv2.imread(path)
            annotated_image = image.copy()
        
            # Convert the BGR image to RGB before processing.
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
            if not results.pose_landmarks:
                print('Not detected')
                continue
    
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    
            base = os.path.basename(path)
            dst_path = 'result_%s' % base
            cv2.imwrite(dst_path, annotated_image)
            print('Detected. Save %s' % dst_path)

if __name__ == '__main__':
    main()
