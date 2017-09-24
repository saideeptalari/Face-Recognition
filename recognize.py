import numpy as np

def recognize_face(embedding, embeddings, labels, threshold=0.5):
    distances = np.linalg.norm(embeddings - embedding, axis=1)
    argmin = np.argmin(distances)
    minDistance = distances[argmin]

    if minDistance>threshold:
        label = "Unknown"
    else:
        label = labels[argmin]

    return (label, minDistance)

if __name__ == "__main__":
    import cv2
    import argparse
    from detectors import detect_faces
    from extractors import extract_face_embeddings
    import cPickle
    import dlib

    ap = argparse.ArgumentParser()
    ap.add_argument("-i","--image", help="Path to image", required=True)
    ap.add_argument("-e","--embeddings", help="Path to saved embeddings",
                    default="face_embeddings.npy")
    ap.add_argument("-l", "--labels", help="Path to saved labels",
                    default="labels.pickle")
    args = vars(ap.parse_args())

    embeddings = np.load(args["embeddings"])
    labels = cPickle.load(open(args["labels"]))
    shape_predictor = dlib.shape_predictor("models/"
                                           "shape_predictor_5_face_landmarks.dat")
    face_recognizer = dlib.face_recognition_model_v1("models/"
                                                     "dlib_face_recognition_resnet_model_v1.dat")

    image = cv2.imread(args["image"])
    image_original = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    faces = detect_faces(image)

    for face in faces:
        embedding = extract_face_embeddings(image, face, shape_predictor, face_recognizer)
        label = recognize_face(embedding, embeddings, labels)
        (x1, y1, x2, y2) = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(image_original, (x1, y1), (x2, y2), (255, 120, 120), 2, cv2.CV_AA)
        cv2.putText(image_original, label[0], (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("Image", image_original)
    cv2.waitKey(0)
