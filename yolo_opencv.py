import sys

from firebase_admin import credentials, firestore, storage
from datetime import datetime
from flask import Flask, request
import optparse
import firebase_admin
import cv2
import numpy as np
import json

# ======================Khai Bao=================================#
app = Flask(__name__)

classes = None

class_ids = []

cred = credentials.Certificate('./Credentials.json')

firebase_admin.initialize_app(cred, {
    'storageBucket': "objectdetection-python.appspot.com"
})

firestore.client()

bucket = storage.bucket()

with open("models_config_label/yolov3.txt", 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))


# ==========================///Khai Bao///=============================#

# ======================Utils Function=================================#
def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


def draw_prediction(index, img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    # label = str(classes[class_id])
    color = COLORS[class_id]
    # cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    print(str(classes[class_id]) + " sub rect: " +
          str(x) + ", " + str(y) + ", " + str(x_plus_w) + ", " + str(y_plus_h))

    cv2.imwrite("./img_detected/big-object-detection.jpg", img)
    cv2.imwrite("./img_detected/object-detection-" + str(index) + ".jpg", img[y: y_plus_h, x:  x_plus_w])
    return "./img_detected/object-detection-" + str(index) + ".jpg"


# ======================///Utils Function///=================================#


# ======================Flask Function=================================#
@app.route("/")
def index():
    return "Hello world!!!"


@app.route('/image_upload', methods=['GET', 'POST'])
def detect():
    path_to_upload = []
    dt = datetime.now()
    file = request.files['image']
    filename = file.filename
    pathStored = "./img_store/" + str(dt.microsecond) + filename

    file.save(pathStored)
    #
    image = cv2.imread(pathStored)
    # image = cv2.imread("/Users/leclev1/Desktop/MU
    # -loan-vi-Alexis-Sanchez-Sao-qua-ta-chieu-menh-Mourinho-sanchez3-1526877991-226-width660height397.jpg")

    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4
    Width = image.shape[1]
    Height = image.shape[0]
    print("Width Height " + str(Width) + ", " + str(Height))
    # scale = 0.00392
    scale = 0.00784

    net = cv2.dnn.readNet("models_config_label/yolov3.weights", "models_config_label/yolov3.cfg")

    blob = cv2.dnn.blobFromImage(image, scale, (608, 608), (0, 0, 0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    boxs = []
    links = []

    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        path_to_upload.append(
            draw_prediction(i, image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h)))
        boxs.append({"x": round(x), "y": round(y), "w": round(x + w), "h": round(y + h)})

    # upload subimages
    for i, path in enumerate(path_to_upload):
        # if i == 0:
        #    blobFirebase = bucket.blob("big_img_detected/" + str(datetime.now().microsecond) + "-detected.jpg")
        #   else:
        blobFirebase = bucket.blob("sub_img_detected/" + str(datetime.now().microsecond) + "-detected.jpg")
        blobFirebase.upload_from_filename(path, content_type='image/jpg')
        blobFirebase.make_public()
        links.append(blobFirebase.public_url)

    # upload big image
    blobFirebaseBigImg = bucket.blob("big_img_detected/" + str(datetime.now().microsecond) + "-detected.jpg")
    blobFirebaseBigImg.upload_from_filename('./img_detected/big-object-detection.jpg', content_type='image/jpg')
    blobFirebaseBigImg.make_public()

    images = [{'sub_img_url': links[v], 'box': boxs[v]} for v in range(0, len(indices))]
    print(json.dumps({"VoImage": {"big_img_url": blobFirebaseBigImg.public_url, "wid": Width, "hei": Height
        , "images": images}}, indent=3))
    return json.dumps({"VoImage": {"big_img_url": blobFirebaseBigImg.public_url, "wid": Width, "hei": Height
        , "images": images}}, indent=3)
    # return send_file("img_detected/big-object-detection.jpg")


# ======================///Flask Function///=================================#


if __name__ == '__main__':
    # app.run(host='192.168.0.19', port=5000)
    parser = optparse.OptionParser(usage="python yolo_opencv.py -p <port>")
    parser.add_option('-p', '--port', action='store', dest='port', help='The port to listen on.')
    (args, _) = parser.parse_args()
    if args.port is None:
        print("Missing required argument: -p/--port")
        sys.exit(1)
    app.run(host='0.0.0.0', port=int(args.port), debug=False)
