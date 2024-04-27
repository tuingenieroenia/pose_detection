import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

with mp_pose.Pose(
    static_image_mode = True ) as pose:
    
    #image = cv2.imread("malo1.jpeg")
    image = cv2.imread("pose2.jpeg")
    height, width, _ = image.shape
    
    imagen_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    results = pose.process(imagen_rgb)
    
    if results.pose_landmarks is not None:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    
cv2.destroyAllWindows()
