# Created by Shengtao Li Based on the following:
#   - find_faces_in_pictures.py
#   - face_distance.py
# 
# updated: 2020-04-15

from PIL import Image
import numpy
import face_recognition

DIR = "/home/shengtao/git/face_recognition/examples"
# Load some images to compare against
ref_image_1 = face_recognition.load_image_file(DIR+"/ref_pics/Zhang-Ziyi.jpg")
ref_image_2 = face_recognition.load_image_file(DIR+"/ref_pics/Fan-Bingbing.jpg")
ref_image_3 = face_recognition.load_image_file(DIR+"/ref_pics/Chi-Ling-Lin.jpg")
ref_image_4 = face_recognition.load_image_file(DIR+"/ref_pics/Gao-Yuanyuan.jpg")
ref_image_5 = face_recognition.load_image_file(DIR+"/ref_pics/Liu-Yifei.jpg")
ref_image_6 = face_recognition.load_image_file(DIR+"/ref_pics/Xu-Jinglei.jpg")
ref_image_7 = face_recognition.load_image_file(DIR+"/ref_pics/Zhang-Jingchu.jpg")
ref_image_8 = face_recognition.load_image_file(DIR+"/ref_pics/Zhang-Yuqi.jpg")
ref_image_9 = face_recognition.load_image_file(DIR+"/ref_pics/Zhao-Wei.jpg")
ref_image_10 = face_recognition.load_image_file(DIR+"/ref_pics/Zhou-Xun.jpg")

# Get the face encodings for the known images
ref1_face_encoding = face_recognition.face_encodings(ref_image_1)[0]
ref2_face_encoding = face_recognition.face_encodings(ref_image_2)[0]
ref3_face_encoding = face_recognition.face_encodings(ref_image_3)[0]
ref4_face_encoding = face_recognition.face_encodings(ref_image_4)[0]
ref5_face_encoding = face_recognition.face_encodings(ref_image_5)[0]
ref6_face_encoding = face_recognition.face_encodings(ref_image_6)[0]
ref7_face_encoding = face_recognition.face_encodings(ref_image_7)[0]
ref8_face_encoding = face_recognition.face_encodings(ref_image_8)[0]
ref9_face_encoding = face_recognition.face_encodings(ref_image_9)[0]
ref10_face_encoding = face_recognition.face_encodings(ref_image_10)[0]

known_encodings = [
    ref1_face_encoding,
    ref2_face_encoding,
    ref3_face_encoding,
    ref4_face_encoding,
    ref5_face_encoding,
    ref6_face_encoding,
    ref7_face_encoding,
    ref8_face_encoding,
    ref9_face_encoding,
    ref10_face_encoding
]

# Load the group picture jpg file into a numpy array
image = face_recognition.load_image_file(DIR+"/unknown_pics/group_pic3.jpg")

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
        best_ref = numpy.where(face_distances == min(face_distances))[0][0]
        best_image = pil_image

    print("face #{} value is {}".format(n, min(face_distances)))

print("The best matching face #{} value is {}".format(best_index, best_image))
print("The ref of best matching face #{} is {}".format(best_index, best_ref))
best_image.show()


