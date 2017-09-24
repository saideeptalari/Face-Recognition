import numpy as np

def extract_face_embeddings(image, face_rect,shape_predictor,face_recognizer):
    shape = shape_predictor(image, face_rect)
    face_embedding = face_recognizer.compute_face_descriptor(image, shape)
    face_embedding = [x for x in face_embedding]
    face_embedding = np.array(face_embedding, dtype="float32")[np.newaxis, :]
    return face_embedding
