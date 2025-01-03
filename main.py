import cv2
from deepface import DeepFace
import time
from threading import Thread
import queue

class EmotionDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.current_emotion = "Initialiazing..."
        self.processing = False
        self.frame_queue = queue.Queue(maxsize=1)

    def analyze_emotion(self):
        while True:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                try:
                    result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                    self.current_emotion = result[0]['dominant_emotion']
                except Exception as e:
                    print(f"Error in emotion detection: {str(e)}")
                self.processing = False
            time.sleep(0.1)

    def run(self):
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            cam = cv2.VideoCapture(1)
        if not cam.isOpened():
            raise IOError("Cannot open webcam")

        emotion_thread = Thread(target=self.analyze_emotion, daemon=True)
        emotion_thread.start()

        last_detection_time = time.time()
        detection_interval = 1.0

        try:
            while True:
                ret,frame = cam.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face = self.face_cascade.detectMultiScale(gray, 1.1, 4)

                for (x,y,w,h) in face:
                    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0),2)

                current_time = time.time()
                if current_time - last_detection_time > detection_interval and not self.processing:
                    self.processing = True
                    last_detection_time = current_time

                    if self.frame_queue.empty():
                        self.frame_queue.put(frame.copy())

                font = cv2.FONT_HERSHEY_COMPLEX
                status_text = "Processing..." if self.processing else self.current_emotion
                cv2.putText(frame, status_text, (50,50), font, 1, (0,0,255), 2, cv2.LINE_4)

                cv2.imshow('Emotion Detection', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cam.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = EmotionDetector()
    detector.run()



