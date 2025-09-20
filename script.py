import os
import sys
import glob
import time
import math
import cv2
import numpy as np
import shutil
import platform
import json
import contextlib
from tqdm import tqdm
from datetime import datetime
from ultralytics import YOLO
import pyttsx3
from threading import Thread

class CombinedDetector:
    def __init__(self, enable_sound=True, calibration_file="camera_calibration_esp.json", notification_cooldown=5.0):
        self.directory = 'data'
        self.COSINE_THRESHOLD = 0.5
        self.temp_unknown_dir = os.path.join(self.directory, 'temp_unknown_faces')
        self.enable_sound = enable_sound

        # Notification cooldown time in seconds
        self.notification_cooldown = notification_cooldown

        # Last notification timestamp - now separate for different entities
        self.last_global_notification_time = 0

        # Last person name spoken - to avoid repeating within the time frame
        self.last_person_announced = None

        # Known widths for distance estimation
        self.KNOWN_WIDTH = {
            'person': 60,
            'car': 180,
            'bottle': 8,
            'chair': 50,
            'laptop': 35,
            'cell phone': 7,
        }
        self.DEFAULT_WIDTH = 30  # cm

        # Load calibration data
        self.load_calibration(calibration_file)

        # Setup components
        if self.enable_sound:
            self.setup_sound()
        self.setup_face_detection()
        self.setup_object_detection()
        self.setup_temp_directory()
        self.load_face_dictionary()
        self.next_unknown_id = 1
        self.detected_unknowns = set()

        # Colors for object visualization
        self.COLORS = np.random.uniform(0, 255, size=(80, 3))

        # Verbose mode flags
        self.verbose_mode = False
        self.model_verbose = False

        # Text-to-speech engine setup
        self.setup_tts()

        # Face tracking for persistent detection
        self.face_tracking = {}  # Dictionary to track faces and their detection time

        # Track all notifications (both faces and objects)
        self.notifications = {
            "faces": {},  # {face_id: last_notification_time}
            "objects": {}  # {object_id: last_notification_time}
        }

        # Track currently visible faces and objects
        self.currently_visible_faces = set()
        self.previously_visible_faces = set()

        # Track currently visible objects
        self.currently_visible_objects = set()
        self.previously_visible_objects = set()

        # Timeout values for reappearance (seconds)
        self.face_reappear_timeout = 3.0
        self.object_reappear_timeout = 3.0

    def setup_tts(self):
        """Setup text-to-speech engine"""
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)  # Speed of speech
        self.tts_busy = False  # Flag to prevent speech overlap
        self.speech_queue = []  # Queue for pending speech

    def speak_text(self, text, force=False):
        """
        Speak text in a non-blocking way

        Args:
            text: Text to speak
            force: If True, speak regardless of cooldown
        """
        current_time = time.time()

        # IMPORTANT: Person detections should always be announced
        is_person_announcement = "person detected" in text.lower()

        # Skip global cooldown for person announcements
        if not force and not is_person_announcement and (current_time - self.last_global_notification_time < self.notification_cooldown):
            if self.verbose_mode:
                print(f"Notification skipped: {text} (cooldown active)")
            return False

        # Add to queue if TTS engine is busy
        if self.tts_busy:
            if self.verbose_mode:
                print(f"Notification queued: {text} (TTS engine busy)")
            self.speech_queue.append(text)
            return True

        def speak_worker():
            self.tts_busy = True
            current_text = text

            while True:
                if self.verbose_mode:
                    print(f"TTS: {current_text}")
                self.tts_engine.say(current_text)
                self.tts_engine.runAndWait()

                # Check if there are more items in the queue
                if self.speech_queue:
                    current_text = self.speech_queue.pop(0)
                else:
                    break

            self.tts_busy = False

        # Only update global notification time for non-person announcements
        if not is_person_announcement:
            self.last_global_notification_time = current_time

        if self.verbose_mode:
            print(f"Speaking notification: {text}")

        tts_thread = Thread(target=speak_worker)
        tts_thread.daemon = True
        tts_thread.start()
        return True

    def setup_sound(self):
        """Setup sound based on platform"""
        self.system = platform.system()
        if self.system == 'Windows':
            try:
                import winsound
                self.sound_function = lambda: winsound.Beep(1000, 500)
                self.sound_available = True
            except Exception as e:
                print(f"Warning: Could not initialize Windows sound: {e}")
                self.sound_available = False
        elif self.system == 'Darwin':
            self.sound_function = lambda: os.system('afplay /System/Library/Sounds/Ping.aiff')
            self.sound_available = True
        elif self.system == 'Linux':
            self.sound_function = lambda: print('\a', flush=True)
            self.sound_available = True
        else:
            print(f"Warning: Sound not supported on {self.system}")
            self.sound_available = False

    def play_notification(self, is_person=False):
        """
        Safely play notification sound if enabled and available

        Args:
            is_person: If True, bypass global cooldown for person notifications
        """
        current_time = time.time()

        # Check global notification cooldown, but bypass for person detections
        if not is_person and current_time - self.last_global_notification_time < self.notification_cooldown:
            if self.verbose_mode:
                print("Sound notification skipped (cooldown active)")
            return False

        if self.enable_sound and self.sound_available:
            try:
                self.sound_function()
                # Update last notification time (only for non-person notifications)
                if not is_person:
                    self.last_global_notification_time = current_time
                return True
            except Exception as e:
                print(f"Warning: Could not play sound: {e}")
                self.sound_available = False
                return False
        return False

    def setup_temp_directory(self):
        """Create or clean temporary directory for unknown faces"""
        if os.path.exists(self.temp_unknown_dir):
            shutil.rmtree(self.temp_unknown_dir)
        os.makedirs(self.temp_unknown_dir)

    def setup_face_detection(self):
        """Initialize face detection and recognition models"""
        weights = os.path.join(self.directory, "models", "face_detection_yunet_2023mar.onnx")
        self.face_detector = cv2.FaceDetectorYN_create(weights, "", (0, 0))
        self.face_detector.setScoreThreshold(0.87)

        weights = os.path.join(self.directory, "models", "face_recognition_sface_2021dec_int8bq.onnx")
        self.face_recognizer = cv2.FaceRecognizerSF_create(weights, "")

    def setup_object_detection(self):
        """Initialize YOLOv10 model for object detection"""
        self.yolo_model = YOLO('yolov10n.pt')

    def load_face_dictionary(self):
        """Load registered faces from image files"""
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
            user_id = os.path.splitext(os.path.basename(file))[0].strip()
            self.dictionary[user_id] = feats[0]

        print(f'Total {len(self.dictionary)} registered IDs loaded')

    def load_calibration(self, filename="camera_calibration.json"):
        """Load camera calibration data from file"""
        try:
            with open(filename, 'r') as f:
                calibration_data = json.load(f)

            self.FOCAL_LENGTH = calibration_data.get("focal_length")
            loaded_widths = calibration_data.get("known_widths", {})
            if loaded_widths:
                self.KNOWN_WIDTH.update(loaded_widths)

            print(f"Calibration loaded from {filename}")
            print(f"Focal length: {self.FOCAL_LENGTH}")
            return True
        except FileNotFoundError:
            print(f"Calibration file {filename} not found.")
            self.FOCAL_LENGTH = None
            return False
        except Exception as e:
            print(f"Error loading calibration: {e}")
            self.FOCAL_LENGTH = None
            return False

    def match(self, feature1):
        """Match face feature against dictionary"""
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
        """Perform face detection and feature extraction"""
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
        """Save unknown face image and features"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_filename = f"unknown_{unknown_id}_{timestamp}.jpg"
        image_path = os.path.join(self.temp_unknown_dir, image_filename)
        cv2.imwrite(image_path, aligned_face)

        feature_filename = f"unknown_{unknown_id}_{timestamp}.npy"
        feature_path = os.path.join(self.temp_unknown_dir, feature_filename)
        np.save(feature_path, features[0])

        return image_path

    def estimate_distance(self, object_width_pixels, class_name):
        """Estimate distance based on object width in pixels"""
        if self.FOCAL_LENGTH is None:
            return None

        known_width = self.KNOWN_WIDTH.get(class_name, self.DEFAULT_WIDTH)
        distance = (known_width * self.FOCAL_LENGTH) / object_width_pixels
        return distance

    def can_notify(self, entity_type, entity_id):
        """
        Check if an entity can be notified based on cooldown rules

        Args:
            entity_type: Either "faces" or "objects"
            entity_id: Unique identifier for the entity

        Returns:
            bool: True if notification is allowed, False otherwise
        """
        current_time = time.time()

        # For faces, always allow notification (we want to announce all detected people)
        if entity_type == "faces":
            return True

        # For objects, check global notification cooldown
        if current_time - self.last_global_notification_time < self.notification_cooldown:
            return False

        # Check specific entity cooldown for objects
        if entity_id in self.notifications[entity_type]:
            last_time = self.notifications[entity_type][entity_id]
            if current_time - last_time < self.notification_cooldown:
                return False

        return True

    def mark_notified(self, entity_type, entity_id):
        """
        Mark an entity as having been notified

        Args:
            entity_type: Either "faces" or "objects"
            entity_id: Unique identifier for the entity
        """
        self.notifications[entity_type][entity_id] = time.time()

        # Only update global notification time for non-face entities
        if entity_type != "faces":
            self.last_global_notification_time = time.time()

    def check_face_persistence(self, face_id, is_known, current_time):
        """
        Check if a face has been consistently detected and is ready for notification

        Returns:
            bool: True if the face should be announced, False otherwise
        """
        # Face tracking format: {id: {'first_seen': timestamp, 'last_seen': timestamp, 'announced': bool}}

        # If face not seen before, add it to tracking
        if face_id not in self.face_tracking:
            self.face_tracking[face_id] = {
                'first_seen': current_time,
                'last_seen': current_time,
                'announced': False,
                'is_known': is_known
            }
            return False

        # Update last seen timestamp
        self.face_tracking[face_id]['last_seen'] = current_time

        # Check if face has been visible for at least 1 second (to avoid false positives)
        time_visible = current_time - self.face_tracking[face_id]['first_seen']

        # If the face has been visible for the minimum time and hasn't been announced yet,
        # or if it reappeared after being gone
        if time_visible >= 1.0:
            return True

        return False

    def cleanup_face_tracking(self, current_time, timeout=3.0):
        """Remove faces that haven't been seen recently"""
        faces_to_remove = []
        for face_id, data in self.face_tracking.items():
            if current_time - data['last_seen'] > timeout:
                faces_to_remove.append(face_id)

        for face_id in faces_to_remove:
            del self.face_tracking[face_id]

    def process_frame(self, frame):
        """Process a frame for face detection/recognition and object detection/distance estimation"""
        start_time = time.time()
        current_time = time.time()
        annotated_frame = frame.copy()

        # Store previously visible entities before updating
        self.previously_visible_faces = self.currently_visible_faces.copy()
        self.previously_visible_objects = self.currently_visible_objects.copy()

        # Clear currently visible lists to rebuild them
        self.currently_visible_faces.clear()
        self.currently_visible_objects.clear()

        # Clean up face tracking data
        self.cleanup_face_tracking(current_time)

        # Face Detection and Recognition
        features, faces, aligned_face = self.recognize_face(frame)
        if faces is not None:
            for idx, (face, feature) in enumerate(zip(faces, features)):
                result, user = self.match(feature)
                box = list(map(int, face[:4]))
                color = (0, 255, 0) if result else (0, 0, 255)
                thickness = 2
                cv2.rectangle(annotated_frame, box, color, thickness, cv2.LINE_AA)

                if result:
                    id_name, score = user
                    is_known = True

                    # Add to currently visible faces
                    self.currently_visible_faces.add(id_name)

                    # Check if this is a new appearance (not in previous frame)
                    is_new_appearance = id_name not in self.previously_visible_faces

                    # Check if face has been consistently detected (to avoid false positives)
                    if is_new_appearance and self.check_face_persistence(id_name, is_known, current_time):
                        # Announce known person with name
                        self.play_notification(is_person=True)
                        self.speak_text(f"Person detected: {id_name}", force=True)
                        self.mark_notified("faces", id_name)
                else:
                    id_name = f"unknown_{self.next_unknown_id}"
                    score = 0.0
                    is_known = False

                    # For unknown faces, we want to use a different tracking approach
                    # since we're generating new IDs for them

                    # Add to currently visible faces using a generic "unknown" marker
                    unknown_marker = "unknown_face"
                    self.currently_visible_faces.add(unknown_marker)

                    if id_name not in self.detected_unknowns:
                        # Check if this unknown face has been consistently detected
                        if self.check_face_persistence(unknown_marker, is_known, current_time):
                            self.play_notification(is_person=True)
                            self.speak_text(f"Unknown person detected", force=True)
                            self.detected_unknowns.add(id_name)
                            self.save_unknown_face(frame, box, aligned_face, features, self.next_unknown_id)
                            self.dictionary[id_name] = feature
                            self.next_unknown_id += 1
                            self.mark_notified("faces", unknown_marker)

                text = f"{id_name} ({score:.2f})"
                position = (box[0], box[1] - 10)
                cv2.putText(annotated_frame, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, color, thickness, cv2.LINE_AA)

        # Object Detection and Distance Estimation
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            if not self.model_verbose:
                with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
                    results = self.yolo_model(frame_rgb, verbose=False)
            else:
                results = self.yolo_model(frame_rgb)
        except Exception as e:
            print(f"Error during object detection: {e}")
            results = []

        for detection in results:
            if len(detection.boxes) == 0:
                continue

            try:
                boxes = detection.boxes.xyxy.cpu().numpy()
                scores = detection.boxes.conf.cpu().numpy()
                class_ids = detection.boxes.cls.cpu().numpy().astype(int)
            except (IndexError, AttributeError):
                continue

            for i, box in enumerate(boxes):
                class_id = class_ids[i]
                class_name = self.yolo_model.names[class_id]
                confidence = scores[i]

                if confidence < 0.5:
                    continue

                x1, y1, x2, y2 = map(int, box)
                object_width_pixels = x2 - x1

                color = tuple(map(int, self.COLORS[class_id % len(self.COLORS)]))
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                distance_text = "Unknown"
                if self.FOCAL_LENGTH is not None:
                    distance = self.estimate_distance(object_width_pixels, class_name)
                    if distance is not None:
                        distance_text = f"{int(distance)} cm"

                label = f"{class_name}: {confidence:.2f}, Dist: {distance_text}"
                label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                y1 = max(y1, label_size[1])

                cv2.rectangle(annotated_frame, (x1, y1 - label_size[1]), (x1 + label_size[0], y1), color, -1)
                cv2.putText(annotated_frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

                # Object announcement logic
                object_id = f"{class_name}_{i}"  # Unique ID for this detection

                # Add to currently visible objects
                self.currently_visible_objects.add(class_name)

                # Check if this is a new appearance
                is_new_appearance = class_name not in self.previously_visible_objects

                # Skip person announcements from YOLO as they'll be handled by face detection
                if class_name != "person":
                    # Only announce on first appearance
                    if is_new_appearance and self.can_notify("objects", class_name):
                        if distance_text != "Unknown":
                            notification_text = f"{class_name} detected at {distance_text}"
                        else:
                            notification_text = f"{class_name} detected"

                        if self.speak_text(notification_text):
                            self.mark_notified("objects", class_name)

        # Display FPS and notification status
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)

        # Display time since last notification (only for objects)
        time_since_notification = current_time - self.last_global_notification_time
        cooldown_status = "READY" if time_since_notification >= self.notification_cooldown else f"{self.notification_cooldown - time_since_notification:.1f}s"
        # cv2.putText(annotated_frame, f"Object Notification: {cooldown_status}", (10, 60),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 255), 2)

        # Display detected faces count
        face_count = len(faces) if faces is not None else 0
        cv2.putText(annotated_frame, f"Faces: {face_count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 2)

        # Display visibility info
        current_faces_text = ", ".join(self.currently_visible_faces) if self.currently_visible_faces else "None"
        if len(current_faces_text) > 40:  # Truncate if too long
            current_faces_text = current_faces_text[:37] + "..."
        cv2.putText(annotated_frame, f"Current faces: {current_faces_text}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 2)

        return annotated_frame

    def toggle_verbose(self):
        """Toggle verbose output mode"""
        self.verbose_mode = not self.verbose_mode
        print(f"Verbose mode {'ON' if self.verbose_mode else 'OFF'}")

    def toggle_model_verbose(self):
        """Toggle model output verbose mode"""
        self.model_verbose = not self.model_verbose
        print(f"Model verbose mode {'ON' if self.model_verbose else 'OFF'}")

    def change_notification_cooldown(self, seconds):
        """Change the notification cooldown period"""
        self.notification_cooldown = max(1.0, float(seconds))
        print(f"Notification cooldown set to {self.notification_cooldown} seconds")

    def cleanup(self):
        """Clean up temporary files and directories"""
        if os.path.exists(self.temp_unknown_dir):
            shutil.rmtree(self.temp_unknown_dir)
            print("Cleaned up temporary unknown faces")

    def run(self):
        """Run the combined detection system"""
        try:
            capture = cv2.VideoCapture(0)
            if not capture.isOpened():
                print("Error: Could not open camera")
                return

            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

            print("=== Combined Detection System (Enhanced) ===")
            print(f"Notification cooldown: {self.notification_cooldown} seconds")
            print("Features:")
            print("- Person and object detection only announced when they first appear")
            print("- Entities must leave frame and return to trigger a new announcement")
            print("- Face recognition with name announcement")
            print("- Object detection with distance estimation")
            print("- Speech queue for multiple announcements")
            print("\nControls:")
            print("Press 'v' to toggle verbose mode")
            print("Press 'm' to toggle model output")
            print("Press '+' to increase cooldown by 1 second")
            print("Press '-' to decrease cooldown by 1 second")
            print("Press 'q' to quit")

            while True:
                ret, frame = capture.read()
                if not ret:
                    break

                processed_frame = self.process_frame(frame)
                processed_frame_enlarged = cv2.resize(processed_frame, (620, 620)) 
                cv2.imshow("Enhanced Detection System", processed_frame_enlarged)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('v'):
                    self.toggle_verbose()
                elif key == ord('m'):
                    self.toggle_model_verbose()
                elif key == ord('+'):
                    self.change_notification_cooldown(self.notification_cooldown + 1.0)
                elif key == ord('-'):
                    self.change_notification_cooldown(self.notification_cooldown - 1.0)

        finally:
            capture.release()
            cv2.destroyAllWindows()
            self.cleanup()

    def run_from_stream(self, stream_url):
        """Run the combined detection system from a video stream"""
        try:
            cap = cv2.VideoCapture(stream_url)

            if not cap.isOpened():
                print("Error: Could not open video stream")
                return

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

            print("Successfully connected to video stream")

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break

                processed_frame = self.process_frame(frame)
                processed_frame_enlarged = cv2.resize(processed_frame, (620, 620)) 
                cv2.imshow("Enhanced Detection System", processed_frame_enlarged)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("\nStopping the stream...")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.cleanup()

def main():
    # Use 5.0 seconds as the notification cooldown period
    print("Initializing Enhanced Detection System...")
    print("Will announce persons and objects only when they first appear or reappear")
    detector = CombinedDetector(enable_sound=True, notification_cooldown=5.0)

    # ESP32-CAM stream URL
    stream_url = "http://192.168.221.200:81/stream"

    # Run from stream
    detector.run_from_stream(stream_url)

if __name__ == '__main__':
    main()
