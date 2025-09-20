import os
import sys
import glob
import time
import math
import cv2 # type: ignore
import numpy as np # type: ignore
import shutil
import platform
from tqdm import tqdm # type: ignore
from datetime import datetime
from ultralytics import YOLO # type: ignore

class CombinedDetector:
    def __init__(self, enable_sound=True):
        self.directory = 'data'
        self.COSINE_THRESHOLD = 0.5
        self.temp_unknown_dir = os.path.join(self.directory, 'temp_unknown_faces')
        self.enable_sound = enable_sound
        
        # Setup sound based on platform
        if self.enable_sound:
            self.setup_sound()

        self.setup_face_detection()
        self.setup_object_detection()
        self.setup_temp_directory()
        self.load_face_dictionary()
        self.next_unknown_id = 1
        self.detected_unknowns = set()

    def setup_sound(self):
        """Setup sound based on platform"""
        self.system = platform.system()
        if self.system == 'Windows':
            try:
                import winsound
                self.sound_function = lambda: winsound.Beep(1000, 500)  # 1000Hz for 500ms
                self.sound_available = True
            except Exception as e:
                print(f"Warning: Could not initialize Windows sound: {e}")
                self.sound_available = False
        elif self.system == 'Darwin':  # macOS
            self.sound_function = lambda: os.system('afplay /System/Library/Sounds/Ping.aiff')
            self.sound_available = True
        elif self.system == 'Linux':
            # Try to use console bell
            self.sound_function = lambda: print('\a', flush=True)
            self.sound_available = True
        else:
            print(f"Warning: Sound not supported on {self.system}")
            self.sound_available = False

    
    def play_notification(self):
        """Safely play notification sound if enabled and available"""
        if self.enable_sound and self.sound_available:
            try:
                self.sound_function()
            except Exception as e:
                print(f"Warning: Could not play sound: {e}")
                self.sound_available = False

    def setup_temp_directory(self):
        if os.path.exists(self.temp_unknown_dir):
            shutil.rmtree(self.temp_unknown_dir)
        os.makedirs(self.temp_unknown_dir)

    def setup_face_detection(self):
        weights = os.path.join(self.directory, "models", "face_detection_yunet_2023mar.onnx")
        self.face_detector = cv2.FaceDetectorYN_create(weights, "", (0, 0))
        self.face_detector.setScoreThreshold(0.87)

        weights = os.path.join(self.directory, "models", "face_recognition_sface_2021dec_int8bq.onnx")
        self.face_recognizer = cv2.FaceRecognizerSF_create(weights, "")

    def setup_object_detection(self):
        self.yolo_model = YOLO('yolov10n.pt')

    def load_face_dictionary(self):
        self.dictionary = {}
        types = ('*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG', '*.JPEG')
        files = []
        for a_type in types:
            files.extend(glob.glob(os.path.join(self.directory, 'images', a_type)))
        files = list(set(files))

        for file in tqdm(files, desc="Loading registered faces"):
            image = cv2.imread(file)
            feats, faces, _ = self.recognize_face(image, file)
            if faces is None:
                continue
            user_id = os.path.splitext(os.path.basename(file))[0]
            self.dictionary[user_id] = feats[0]

        print(f'Total {len(self.dictionary)} registered IDs loaded')

    def match(self, feature1):
        max_score = 0.0
        sim_user_id = ""
        for user_id, feature2 in zip(self.dictionary.keys(), self.dictionary.values()):
            score = self.face_recognizer.match(
                feature1, feature2, cv2.FaceRecognizerSF_FR_COSINE)
            if score >= max_score:
                max_score = score
                sim_user_id = user_id
        if max_score < self.COSINE_THRESHOLD:
            return False, ("", 0.0)
        return True, (sim_user_id, max_score)

    def recognize_face(self, image, file_name=None):
        channels = 1 if len(image.shape) == 2 else image.shape[2]
        if channels == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if channels == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        if image.shape[0] > 1000:
            image = cv2.resize(image, (0, 0),
                             fx=500 / image.shape[0], fy=500 / image.shape[0])

        height, width, _ = image.shape
        self.face_detector.setInputSize((width, height))
        try:
            _, faces = self.face_detector.detect(image)
            if file_name is not None:
                assert len(faces) > 0, f'the file {file_name} has no face'

            faces = faces if faces is not None else []
            features = []
            aligned_face = None
            for face in faces:
                aligned_face = self.face_recognizer.alignCrop(image, face)
                feat = self.face_recognizer.feature(aligned_face)
                features.append(feat)
            return features, faces, aligned_face
        except Exception as e:
            print(e)
            print(file_name)
            return None, None, None

    def save_unknown_face(self, image, face_box, aligned_face, features, unknown_id):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_filename = f"unknown_{unknown_id}_{timestamp}.jpg"
        image_path = os.path.join(self.temp_unknown_dir, image_filename)
        cv2.imwrite(image_path, aligned_face)
        
        feature_filename = f"unknown_{unknown_id}_{timestamp}.npy"
        feature_path = os.path.join(self.temp_unknown_dir, feature_filename)
        np.save(feature_path, features[0])
        
        return image_path

    def process_frame(self, frame):
        start_time = time.time()
        
        # Face Detection and Recognition
        features, faces, aligned_face = self.recognize_face(frame)
        if faces is not None:
            for idx, (face, feature) in enumerate(zip(faces, features)):
                result, user = self.match(feature)
                box = list(map(int, face[:4]))
                color = (0, 255, 0) if result else (0, 0, 255)
                thickness = 2
                cv2.rectangle(frame, box, color, thickness, cv2.LINE_AA)

                if result:
                    id_name, score = user
                else:
                    id_name = f"unknown_{self.next_unknown_id}"
                    score = 0.0
                    if id_name not in self.detected_unknowns:
                        self.play_notification()
                        self.detected_unknowns.add(id_name)
                        self.save_unknown_face(frame, box, aligned_face, features, self.next_unknown_id)
                        self.dictionary[id_name] = feature
                        self.next_unknown_id += 1

                text = f"{id_name} ({score:.2f})"
                position = (box[0], box[1] - 10)
                cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, color, thickness, cv2.LINE_AA)

        # Object Detection
        results = self.yolo_model.predict(source=frame)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = result.names[int(box.cls[0])]
                confidence = box.conf[0]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                text = f"{label}: {confidence:.2f}"
                cv2.putText(frame, text, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame

    def cleanup(self):
        """Clean up temporary files and directories"""
        if os.path.exists(self.temp_unknown_dir):
            shutil.rmtree(self.temp_unknown_dir)
            print("Cleaned up temporary unknown faces")

    def run(self):
        try:
            capture = cv2.VideoCapture(0)
            if not capture.isOpened():
                print("Error: Could not open camera")
                return

            while True:
                ret, frame = capture.read()
                if not ret:
                    break

                processed_frame = self.process_frame(frame)
                cv2.imshow("Combined Detection", processed_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            capture.release()
            cv2.destroyAllWindows()
            self.cleanup()

def main():
    # Initialize with sound disabled by default
    detector = CombinedDetector(enable_sound=True)
    detector.run()

if __name__ == '__main__':
    main()


#detector = CombinedDetector(enable_sound=False)
