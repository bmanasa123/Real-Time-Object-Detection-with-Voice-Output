import cv2
import pyttsx3
import threading
import time
#OpenCV DNN

net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights","dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1/255)
#Load class lists
classes = []
with open("dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

#Initialize camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1500)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 700)



#Create window
cv2.namedWindow("Frame")

engine = pyttsx3.init()
engine.setProperty("rate",150)
engine_lock = threading.Lock()
detected_objects = set()
speak_flag = True

#FULL HD 1920 x 1080
def click_button(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)

def speak(detected_objects) :
    announcement = f"There is a {', '.join(detected_objects)} in front of you."
    with engine_lock:
        engine.say(announcement)
        engine.runAndWait()

def speak_thread() :
    while True:
        if speak_flag:
            speak(detected_objects)
            time.sleep(0.5)
speech_thread = threading.Thread(target=speak_thread)
speech_thread.start()
while True:
    #Get frames
    ret, frame = cap.read()
    #Object Detection

    (class_ids, scores, bboxes) = model.detect(frame)
    detected_objects.clear()
    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        (x, y, w, h) = bbox
        class_name = classes[class_id]

        cv2.putText(frame, str(class_name), (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (200, 0, 50), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 0, 50), 3)
        detected_objects.add(class_name)

        #print("class ids", class_ids)
        #print("Scores", scores)
        #print("bboxes",bboxes)

    #create a button
    cv2.rectangle(frame, (20, 20), (300, 70), (0, 0, 200), -1)
    #polygon = [(20, 20), (220,20), (220,70), (20, 70)]
    #cv2.fillPoly(frame, polygon, (0, 0, 200))
    cv2.putText(frame, class_name , (30, 60), cv2.FONT_HERSHEY_PLAIN, 3, (225, 225, 225), 3)

    cv2.imshow("Frame", frame)
    cv2.waitKey(2)
cap.release()
cv2.destroyAllWindows()















