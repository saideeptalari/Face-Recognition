from extractors import extract_face_embeddings
from detectors import detect_faces
from db import add_embeddings
import dlib

shape_predictor = dlib.shape_predictor("models/shape_predictor_5_face_landmarks.dat")
face_recognizer = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")

def enroll_face(image, label,
                embeddings_path="face_embeddings.npy",
                labels_path="labels.pickle", down_scale=1.0):

    faces = detect_faces(image, down_scale)
    if len(faces)<1:
        return False
    if len(faces)>1:
        raise ValueError("Multiple faces not allowed for enrolling")
    face = faces[0]
    face_embeddings = extract_face_embeddings(image, face, shape_predictor,
                                              face_recognizer)
    add_embeddings(face_embeddings, label, embeddings_path=embeddings_path,
                   labels_path=labels_path)
    return True

if __name__ == "__main__":
    import cv2
    import glob
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("-d","--dataset", help="Path to dataset to enroll", required=True)
    ap.add_argument("-e","--embeddings", help="Path to save embeddings",
                    default="face_embeddings.npy")
    ap.add_argument("-l","--labels", help="Path to save labels",
                    default="labels.cpickle")

    args = vars(ap.parse_args())
    filetypes = ["png", "jpg"]
    dataset = args["dataset"].rstrip("/")
    imPaths = []

    for filetype in filetypes:
        imPaths += glob.glob("{}/*/*.{}".format(dataset, filetype))

    for path in imPaths:
        label = path.split("/")[-2]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        enroll_face(image, label, embeddings_path=args["embeddings"],
                    labels_path=args["labels"])