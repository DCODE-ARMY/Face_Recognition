from tabnanny import verbose
import numpy as np
import cv2
from PIL import Image
from keras_facenet import FaceNet
from mtcnn import MTCNN
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cosine
import threading
import time

class Face_Recognition:
    def __init__(self):
        self.embedder = FaceNet()  # Ensure this is using TensorFlow GPU
        self.detector = MTCNN()    # Ensure this is using TensorFlow GPU
        self.frame = None
        self.face_array = None
        self.box = None
        self.processing = False
        self.faces_data=[]

    def extract_face(self, pixels, required_size=(160, 160)):
        results = self.detector.detect_faces(pixels)
        if not results:
            return None, None

        x1, y1, width, height = results[0]['box']
        x2, y2 = x1 + width, y1 + height
        face = pixels[y1:y2, x1:x2]
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = np.asarray(image)
        return face_array, (x1, y1, x2, y2)
 
    def find_closest_match(self, face_embedding, known_embeddings, known_labels):
        similarities = [1 - cosine(face_embedding, known_embedding) for known_embedding in known_embeddings]
        best_match_index = np.argmax(similarities)
        best_match_label = known_labels[best_match_index]
        best_similarity = similarities[best_match_index]
        return best_match_label, best_similarity

    def detect_faces_thread(self, cap, known_embeddings, known_labels):
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            self.frame = cv2.resize(frame, (640, 480))
            pixels = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            results = self.detector.detect_faces(pixels,verbose=0)
            self.faces_data=[]
            for result in results:
                x1, y1, width, height = result['box']
                x2, y2 = x1 + width, y1 + height
                face_array = pixels[y1:y2, x1:x2]
                self.face_array = face_array
                self.box = (x1, y1, x2, y2)
                self.faces_data.append((face_array, (x1, y1, x2, y2)))

            #time.sleep(5)  # Adjust as necessary

    def extract_embeddings_thread(self, known_embeddings, known_labels):
        while True:
            if self.frame is not None:
                frame_copy = self.frame.copy()  # Make a copy to avoid modifying the original frame concurrently
                for face_array, (x1, y1, x2, y2) in self.faces_data:
                    face_embedding = self.embedder.embeddings(np.expand_dims(face_array, axis=0),verbose=0)[0]
                    face_embedding = normalize([face_embedding])[0]

                    best_match_label, best_similarity = self.find_closest_match(face_embedding, known_embeddings, known_labels)
                    label = f'{best_match_label}: {best_similarity:.2f}' if best_similarity > 0.5 else 'Not Matched'
                    color = (0, 255, 0) if best_similarity > 0.5 else (0, 0, 255)
                
                #     cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
                #     cv2.putText(frame_copy, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                # cv2.imshow('Webcam Face Recognition', frame_copy)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def start_processing(self, cap, known_embeddings, known_labels):
        threading.Thread(target=self.detect_faces_thread, args=(cap, known_embeddings, known_labels), daemon=True).start()
        threading.Thread(target=self.extract_embeddings_thread, args=(known_embeddings, known_labels), daemon=True).start()

        
    def test(self):
        print('testing if we can work with this')

# if __name__ == '__main__':
#     face_recognition = Face_Recognition()

#     known_faces = [
#         r'C:\Users\darsh\source\repos\Face_Recognition\Face_Recognition\im_1.jpg',
#         r'C:\Users\darsh\source\repos\Face_Recognition\Face_Recognition\im_4.jpg',
#         r'C:\Users\darsh\source\repos\Face_Recognition\Face_Recognition\im_7.jpg',
#         r'C:\Users\darsh\source\repos\Face_Recognition\Face_Recognition\img_11.jpg'
#     ]
#     known_labels = ['raina', 'dhoni', 'kholi', 'darshan']

#     known_embeddings = []
#     for face_path in known_faces:
#         face_img = np.asarray(Image.open(face_path).convert('RGB'))
#         face_array, _ = face_recognition.extract_face(face_img)
#         if face_array is not None:
#             face_embedding = face_recognition.embedder.embeddings(np.expand_dims(face_array, axis=0))[0]
#             known_embeddings.append(normalize([face_embedding])[0])

#     cap = cv2.VideoCapture(0)
#     face_recognition.start_processing(cap, known_embeddings, known_labels)

#     while True:
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()
