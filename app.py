from flask import Flask, render_template, Response
import cv2
import threading
import pyttsx3

app = Flask(__name__)

net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1 / 255)

classes = []
with open("dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1050)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

engine = pyttsx3.init()
engine.setProperty("rate", 150)
engine_lock = threading.Lock()
detected_objects = set()
speak_flag = True


def click_button(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)


def speak(detected_objects):
    announcement = ', '.join(detected_objects)
    with engine_lock:
        engine.say(announcement)
        engine.runAndWait()


def speak_thread():
    while True:
        if speak_flag:
            speak(detected_objects)
            time.sleep(0.5)


speech_thread = threading.Thread(target=speak_thread)
speech_thread.start()


def detect_objects():
    while True:
        ret, frame = cap.read()

        (class_ids, scores, bboxes) = model.detect(frame)

        detected_objects.clear()
        for class_id, score, bbox in zip(class_ids, scores, bboxes):
            (x, y, w, h) = bbox
            class_name = classes[class_id]

            cv2.putText(frame, str(class_name), (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (200, 0, 50), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 0, 50), 3)
            detected_objects.add(class_name)

        _, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')


@app.route('/')
def index():
    return render_template('index.html')


def gen():
    while True:
        frame = cap.read()
        _, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(detect_objects(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
