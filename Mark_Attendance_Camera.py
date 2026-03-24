import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet

detector = MTCNN()
facenet = FaceNet()

attendance_mean_embedding = None

def capture_attendance_face_streamlit(max_faces=5):

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    try:
        while True:

            captured_embeddings = []
            captured_keypoints = []
            frame_count = 0

            while frame_count < max_faces:

                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # CRITICAL PROTECTION (MTCNN crash fix)

                try:
                    faces = detector.detect_faces(rgb_frame)
                except Exception as e:
                    print("Detection error:", e)
                    yield rgb_frame, None, None, None
                    continue

                # No face → skip safely

                if not faces:
                    yield rgb_frame, None, None, None
                    continue

                face = faces[0]
                x, y, w, h = face['box']

                # STRICT VALIDATION

                if w < 50 or h < 50:
                    yield rgb_frame, None, None, None
                    continue

                x, y = max(0, x), max(0, y)

                # Ensure within bounds

                if y + h > rgb_frame.shape[0] or x + w > rgb_frame.shape[1]:
                    yield rgb_frame, None, None, None
                    continue

                face_crop = rgb_frame[y:y+h, x:x+w]

                #  Empty or invalid crop

                if face_crop is None or face_crop.size == 0:
                    yield rgb_frame, None, None, None
                    continue

                # Resize to safe size

                try:
                    face_crop = cv2.resize(face_crop, (160, 160))
                except:
                    yield rgb_frame, None, None, None
                    continue

                # FINAL PROTECTION (MODEL CRASH FIX)
                try:
                    embedding = facenet.embeddings([face_crop])[0]
                    embedding = np.array(embedding, dtype=np.float32)
                except Exception as e:
                    print("Embedding error:", e)
                    yield rgb_frame, None, None, None
                    continue

                captured_embeddings.append(embedding)
                captured_keypoints.append(face['keypoints'])

                frame_count += 1

                # Draw box
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Draw keypoints
                for point in face['keypoints'].values():
                    px, py = point
                    cv2.circle(frame, (px, py), 3, (0, 0, 255), -1)

                yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), None, None, None

            # After collecting frames
            
            if len(captured_embeddings) == max_faces:

                mean_embedding = np.mean(captured_embeddings, axis=0)

                global attendance_mean_embedding
                attendance_mean_embedding = mean_embedding

                yield None, captured_embeddings, captured_keypoints, mean_embedding

    finally:
        cap.release()