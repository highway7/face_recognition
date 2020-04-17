# Created by Shengtao Li Based on the following:
#   - find_faces_in_pictures.py
#   - face_distance.py
# 
# updated: 2020-04-15

from PIL import Image
import face_recognition

from tempfile import NamedTemporaryFile
from shutil import copyfileobj
from os import remove
from flask import Flask, jsonify, request, redirect, send_file
import io

# You can change this to any folder on your system
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_image():
    # Check if a valid image file was uploaded
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # The image file seems valid! Detect faces and return the result.
            return find_and_rank(file)

    # If no valid image file was uploaded, show the file upload form:
    return '''
    <!doctype html>
    <title>Face Value Analyzer</title>
    <h1>Upload a group picture and see whose looks close to a celebrity!</h1>
    <form method="POST" enctype="multipart/form-data">
      <input type="file" name="file">
      <input type="submit" value="Upload">
    </form>
    '''

def find_and_rank(file_stream):
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
    image = face_recognition.load_image_file(file_stream)

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

 
    output = io.BytesIO()
    best_image.convert('RGBA').save(output, format='PNG')
    output.seek(0, 0)
    return send_file(output, mimetype='image/png', as_attachment=False)    

    #tempFileObj = NamedTemporaryFile(mode='w+b',suffix='jpg')
    #copyfileobj(best_image,tempFileObj)
    #tempFileObj.seek(0,0)
    
    #print("The best matching face #{} value is {}".format(best_index, best_image))
    #best_image.show()
    #response = send_file(tempFileObj, as_attachment=True, attachment_filename='myfile.jpg')
    #return response



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)