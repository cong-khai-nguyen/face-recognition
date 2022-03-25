import cmake
import os
import face_recognition as fr
import dlib
import numpy as np
from time import sleep
import cv2

# function to encodes all the faces in the faces folder
# return a dict of (name, image encoded) -> Ex: bill gate : [......]
def get_encoded_faces():
    encoded = {}

    for dirpath, dnames, file_names in os.walk("./media/faces"):
        # print(dnames)
        # print(dirpath)
        # print(file_names)
        for f in file_names:
            if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".jpeg"):
                face = fr.load_image_file("media/faces/" + f)
                # print(fr.face_encodings(face)[0])
                # print("\n")
                encoding = fr.face_encodings(face)[0]
                encoded[f.split(".")[0]] = encoding
    return encoded

# function to encode a face give the file name
def unknown_image_encoded(img):
    face = fr.load_image_file("media/faces/" + img)
    encoding = fr.face_encodings(face)[0]

    return encoding

def classify_face(im):
    # Get every face image I have encoded
    faces = get_encoded_faces()
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())

    img = cv2.imread(im, 1)
    img = cv2.resize(img, (0,0), fx = 0.5, fy =0.5)

    face_locations = fr.face_locations(img)
    unknown_face_encodings = fr.face_encodings(img, face_locations)

    face_names = []
    for face_encoding in unknown_face_encodings:
        # See if the face is a match in any of known faces
        matches = fr.compare_faces(faces_encoded, face_encoding)
        name = "Unknown"

        # Use the known face with the smallest distance to the new face
        face_distances = fr.face_distance(faces_encoded, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw a box around the face
            cv2.rectangle(img, (left-20, top-20), (right+20, bottom+20), (255, 0, 0), 2)

            #Draw label
            cv2.rectangle(img, (left-20, bottom-15), (right+20, bottom+20), (255,0,0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, name, (left-20, bottom+15), font, 1.0, (255,255,255), 2)

    while True:
        cv2.imshow('Video', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return face_names



classify_face("test.jpg")