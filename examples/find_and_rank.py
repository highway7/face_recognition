# Created by Shengtao Li Based on the following:
#   - find_faces_in_pictures.py
#   - face_distance.py
# 
# updated: 2020-04-15

from PIL import Image
import face_recognition

DIR = "/home/shengtao/git/face_recognition/examples"
# Load some images to compare against
ref_image_1 = face_recognition.load_image_file(DIR+"/ref_pics/Zhang-Ziyi.jpg")
ref_image_2 = face_recognition.load_image_file(DIR+"/ref_pics/Fan-Bingbing.jpg")

# Get the face encodings for the known images
ref1_face_encoding = face_recognition.face_encodings(ref_image_1)[0]
ref2_face_encoding = face_recognition.face_encodings(ref_image_2)[0]

known_encodings = [
    ref1_face_encoding,
    ref2_face_encoding
]

# Load the group picture jpg file into a numpy array
image = face_recognition.load_image_file(DIR+"/unknown_pics/pic_1.jpg")

# Find all the faces in the image using the default HOG-based model.
# This method is fairly accurate, but not as accurate as the CNN model and not GPU accelerated.
# See also: find_faces_in_picture_cnn.py
face_locations = face_recognition.face_locations(image)

print("I found {} face(s) in this photograph.".format(len(face_locations)))

best_dist = 1.00
n = 0

for face_location in face_locations:

    # Print the location of each face in this image
    top, right, bottom, left = face_location
    ## print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

    # You can access the actual face itself like this:
    face_image = image[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)
    #pil_image.show()

    image_to_test_encoding = face_recognition.face_encodings(image,{face_location},1)

    #image_to_test_encoding = face_recognition.face_encodings(face_image)[0]
    # See how far apart the test image is from the known faces
    face_distances = face_recognition.face_distance(known_encodings, image_to_test_encoding[0])
    
    n = n + 1
    if min(face_distances) < best_dist:
        best_dist = min(face_distances)
        best_index = n
        best_image = pil_image
    
    print("face #{} value is {}".format(n, min(face_distances)))

print("The best matching face #{} value is {}".format(best_index, best_image))
best_image.show()
