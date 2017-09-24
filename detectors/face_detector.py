import dlib
import cv2

face_detector = dlib.get_frontal_face_detector()
#face_detector = dlib.cnn_face_detection_model_v1("models/mmod_human_face_detector.dat")

def scale_faces(face_rects, down_scale=1.5):
    faces = []
    for face in face_rects:
        scaled_face = dlib.rectangle(int(face.left() * down_scale),
                                    int(face.top() * down_scale),
                                    int(face.right() * down_scale),
                                    int(face.bottom() * down_scale))
        faces.append(scaled_face)
    return faces

def detect_faces(image, down_scale=1.5):
    image_scaled = cv2.resize(image, None, fx=1.0/down_scale, fy=1.0/down_scale,
                              interpolation=cv2.INTER_LINEAR)
    faces = face_detector(image_scaled, 0)
  #  faces = [face.rect for face in faces]
    faces = scale_faces(faces, down_scale)
    return faces

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", help="Path to image", required=True)
    args = vars(ap.parse_args())

    image = cv2.imread(args["image"])
    faces = detect_faces(image, down_scale=0.5)
    
    for face in faces:
        x,y,w,h = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(image, (x,y), (w,h), (255,200,150), 2, cv2.CV_AA)

    cv2.imshow("Image", image)
    cv2.waitKey(0)
