import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet

# Initialize detector and FaceNet once

detector = MTCNN()
facenet = FaceNet()

def capture_faces_streamlit(max_faces=10):
    """
    Generator to capture 10 frames and create embeddings.
    Yields:
        frame: RGB frame to display
        embeddings: list of dicts with 'embedding' and 'keypoints'
        mean_embedding: numpy array of mean embedding (None until ready)
    """
    cap = cv2.VideoCapture(0) 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    captured_embeddings = []
    mean_embedding = None
    frame_count = 0

    try:
        while frame_count < max_faces:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Protect MTCNN crash
            try:
                faces = detector.detect_faces(rgb_frame)
            except Exception as e:
                print("Detection error:", e)
                yield rgb_frame, [], None
                continue

            if len(faces) == 0:
                yield rgb_frame, [], None
                continue

            if faces:

                # Take only first detected face per frame

                face = faces[0]
                x, y, w, h = face['box']
                x, y = max(0, x), max(0, y)

                # Skip small faces

                if w < 50 or h < 50:
                    yield rgb_frame, [], None
                    continue

                # Ensure inside frame

                if y + h > rgb_frame.shape[0] or x + w > rgb_frame.shape[1]:
                    yield rgb_frame, [], None
                    continue

                face_crop = rgb_frame[y:y+h, x:x+w]

                # Empty crop protection

                if face_crop is None or face_crop.size == 0:
                    yield rgb_frame, [], None
                    continue

                # Resize (VERY IMPORTANT)
                
                try:
                    face_crop = cv2.resize(face_crop, (160, 160))
                except:
                    yield rgb_frame, [], None
                    continue

                # Safe embedding

                try:
                    embedding = facenet.embeddings([face_crop])[0]
                    embedding = np.array(embedding, dtype=np.float32)
                except Exception as e:
                    print("Embedding error:", e)
                    yield rgb_frame, [], None
                    continue

                captured_embeddings.append({
                    'embedding': embedding,
                    'keypoints': face['keypoints']
                })

                frame_count += 1

                # Draw rectangle

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Draw keypoints

                for point in face['keypoints'].values():
                    cv2.circle(frame, point, 2, (0, 0, 255), -1)

            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), captured_embeddings, mean_embedding

        # Compute mean embedding after 10 captures
        
        if len(captured_embeddings) == max_faces:
            all_embeddings = [item['embedding'] for item in captured_embeddings]
            mean_embedding = np.mean(all_embeddings, axis=0)

            yield None, captured_embeddings, mean_embedding

    finally:
        cap.release()