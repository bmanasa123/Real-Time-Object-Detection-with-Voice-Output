import cv2
import pyttsx3
import threading
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
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1050)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)


#FULL HD 1920 x 1080
def click_button(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)




#Create window
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", click_button)

engine = pyttsx3.init()
engine.setProperty("rate",130)

engine_lock = threading.Lock()

def speak(class_name) :
    announcement = f"{class_name}"
    with engine_lock:
        engine.say(announcement)
        engine.runAndWait()
        engine.stop()
def speak_thread(class_name) :
    t = threading.Thread(target=speak, args=(class_name,))
    t.start()

while True:
    #Get frames
    ret, frame = cap.read()
    #Object Detection


    (class_ids, scores, bboxes) = model.detect(frame)
    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        (x, y, w, h) = bbox
        class_name = classes[class_id]

        cv2.putText(frame, str(class_name), (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (200, 0, 50), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 0, 50), 3)

        speak_thread(class_name)
        '''t = threading.Thread(target=speak, args=(class_name,))
        t.start()'''
        '''engine.say(class_name)
        engine.runAndWait()
        engine.stop()'''



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










