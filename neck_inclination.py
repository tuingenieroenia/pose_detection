#Importamos la bibliotecas necesarias
import cv2
import time
import math as m
import mediapipe as mp

# Inicializamos los elementos de mediapipe
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic

#Definimos la función que calcula la distancia entre los dos hombros
def findDistance(x1, y1, x2, y2):
    dist = m.sqrt((x2-x1)**2+(y2-y1)**2)
    return dist

#Definimos la función que calcula el ángulo de inclinación
def findAngle(x1, y1, x2, y2):
    theta = m.acos((y2 -y1)*(-y1) / (m.sqrt((x2 - x1)**2 + (y2 - y1)**2) * y1))
    degree = int(180/m.pi)*theta
    return degree

#Definimos la función que envía una alerta
def sendWarning(x):
    pass

#Inicializamos los contadores de buenos y malos frames
good_frames = 0
bad_frames = 0

#Definimos la fuente de los mensajes
font = cv2.FONT_HERSHEY_SIMPLEX

#Definimos los colores a utilizar en el proyecto
blue = (255, 127, 0)
red = (50, 50, 255)
green = (127, 255, 0)
dark_blue = (127, 20, 0)
light_green = (127, 233, 100)
yellow = (0, 255, 255)
pink = (255, 0, 255)

#Inicializamos la clase de captura de pose de mediapipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

#Iniciamos la captura de video
#Archivo de entrada
#file_name = 'input.mp4' 
#cap = cv2.VideoCapture(file_name)

#Para video en vivo
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


#Definimos la metadata
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_size = (width, height)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

#Iniciamos el escritor de video de OpenCV
video_output = cv2.VideoWriter('output.mp4', fourcc, fps, frame_size)

#Iniciamos el procesamiento
print('Processing..')
#Mientras la captura de video este en ejecución
while cap.isOpened():
    
    #Capturamos los frames
    success, image = cap.read()
    if not success:
        print("Null.Frames")
        break
    
    #Obtenemos los fotogramas por segundo
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    #Obtenemos el alto y ancho de la imagen
    h, w = image.shape[:2]

    #MediaPipe trabaja con imagenes en RGB y OpenCV con BGR así que convertimos
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #Obtenemos los 33 puntos que identifica mediapipe 
    keypoints = pose.process(image)

    #Volvemos a convertir la imagen a BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    #Asignamos los elementos a variables para poder manipularlos
    lm = keypoints.pose_landmarks
    lmPose = mp_pose.PoseLandmark

    # Obtenemos las coordenadas de los puntos clave
    # Once aligned properly, left or right should not be a concern.      
    # Hombro izquierdo
    l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
    l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
    # Hombro derecho
    r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
    r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)
    # Oreja izquierda
    l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * w)
    l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * h)
    # Cadera izquierda
    l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
    l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)

    #Calculamos la distancia entre el hombro izquierdo y el hombro derecho
    offset = findDistance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)

    # Assist to align the camera to point at the side view of the person.
    # Offset threshold 30 is based on results obtained from analysis over 100 samples.
    if offset < 100:
        cv2.putText(image, str(int(offset)) + ' Aligned', (w - 150, 30), font, 0.9, green, 2)
    else:
        cv2.putText(image, str(int(offset)) + ' Not Aligned', (w - 150, 30), font, 0.9, red, 2)

    #Se calculan los angulos de inclinación
    neck_inclination = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
    torso_inclination = findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)

    #Dibujamos las lineas de referencia
    cv2.circle(image, (l_shldr_x, l_shldr_y), 7, yellow, -1)
    cv2.circle(image, (l_ear_x, l_ear_y), 7, yellow, -1)

    # Let's take y - coordinate of P3 100px above x1,  for display elegance.
    # Although we are taking y = 0 while calculating angle between P1,P2,P3.
    cv2.circle(image, (l_shldr_x, l_shldr_y - 100), 7, yellow, -1)
    cv2.circle(image, (r_shldr_x, r_shldr_y), 7, pink, -1)
    cv2.circle(image, (l_hip_x, l_hip_y), 7, yellow, -1)

    # Similarly, here we are taking y - coordinate 100px above x1. Note that
    # you can take any value for y, not necessarily 100 or 200 pixels.
    cv2.circle(image, (l_hip_x, l_hip_y - 100), 7, yellow, -1)

    # Put text, Posture and angle inclination.
    # Text string for display.
    angle_text_string = 'Neck : ' + str(int(neck_inclination)) + '  Torso : ' + str(int(torso_inclination))

    # Determine whether good posture or bad posture.
    # The threshold angles have been set based on intuition.
    if neck_inclination < 40 and torso_inclination < 20:
        bad_frames = 0
        good_frames += 1

        cv2.putText(image, angle_text_string, (10, 30), font, 0.9, light_green, 2)
        cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9, light_green, 2)
        cv2.putText(image, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), font, 0.9, light_green, 2)

        # Join landmarks.
        cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), green, 4)
        cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), green, 4)
        cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), green, 4)
        cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), green, 4)

    else:
        good_frames = 0
        bad_frames += 1

        cv2.putText(image, angle_text_string, (10, 30), font, 0.9, red, 2)
        cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9, red, 2)
        cv2.putText(image, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), font, 0.9, red, 2)

        # Join landmarks.
        cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), red, 4)
        cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), red, 4)
        cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), red, 4)
        cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), red, 4)

    # Calculate the time of remaining in a particular posture.
    good_time = (1 / fps) * good_frames
    bad_time =  (1 / fps) * bad_frames

    # Pose time.
    if good_time > 0:
        time_string_good = 'Good Posture Time : ' + str(round(good_time, 1)) + 's'
        cv2.putText(image, time_string_good, (10, h - 20), font, 0.9, green, 2)
    else:
        time_string_bad = 'Bad Posture Time : ' + str(round(bad_time, 1)) + 's'
        cv2.putText(image, time_string_bad, (10, h - 20), font, 0.9, red, 2)

    # If you stay in bad posture for more than 3 minutes (180s) send an alert.
    if bad_time > 180:
        sendWarning()
    # Write frames.
    video_output.write(image)
    
print('Finished.')
cap.release()
cv2.destroyAllWindows()
video_output.release()