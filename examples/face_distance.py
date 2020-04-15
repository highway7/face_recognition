import face_recognition
from PIL import Image

# Often instead of just checking if two faces match or not (True or False), it's helpful to see how similar they are.
# You can do that by using the face_distance function.

# The model was trained in a way that faces with a distance of 0.6 or less should be a match. But if you want to
# be more strict, you can look for a smaller face distance. For example, using a 0.55 cutoff would reduce false
# positive matches at the risk of more false negatives.

# Note: This isn't exactly the same as a "percent match". The scale isn't linear. But you can assume that images with a
# smaller distance are more similar to each other than ones with a larger distance.

# Load some images to compare against
ref_image_1 = face_recognition.load_image_file("./ref_pics/Zhang-Ziyi.jpg")
ref_image_2 = face_recognition.load_image_file("./ref_pics/Fan-Bingbing.jpg")

# Get the face encodings for the known images
ref1_face_encoding = face_recognition.face_encodings(ref_image_1)[0]
ref2_face_encoding = face_recognition.face_encodings(ref_image_2)[0]

known_encodings = [
    ref1_face_encoding,
    ref2_face_encoding
]

# Load a test image and get encondings for it
image_to_test = face_recognition.load_image_file("./ref_pics/Zhang-Ziyi.jpg")
image_to_test_encoding = face_recognition.face_encodings(image_to_test)[0]

# See how far apart the test image is from the known faces
face_distances = face_recognition.face_distance(known_encodings, image_to_test_encoding)


for i, face_distance in enumerate(face_distances):
    print("The test image has a distance of {:.2} from known image #{}".format(face_distance, i))
    print("- With a normal cutoff of 0.6, would the test image match the known image? {}".format(face_distance < 0.6))
    print("- With a very strict cutoff of 0.5, would the test image match the known image? {}".format(face_distance < 0.5))
    print()
