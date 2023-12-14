from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import dlib
import os
from datetime import datetime
from detector import DrowsinessDetector

app = Flask(__name__)
detector = DrowsinessDetector()

# Set up directories for storing files
VIDEO_DIR = 'storage/video'
IMAGE_DIR = 'storage/image'
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

@app.route('/')
def index():
    return "WELCOME TO OUR DROWSINESS DETECTOR"

@app.route('/detect_drowsiness', methods=['POST'])
def detect_drowsiness():
    response = {"drowsy": False}

    try:
        # Save file with timestamped filename
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")
        video_filename = os.path.join(VIDEO_DIR, f"{current_time}.mp4")
        image_filename = os.path.join(IMAGE_DIR, f"{current_time}.jpg")

        file = request.files['file']
        file.save(video_filename)

        vidObj = cv2.VideoCapture(video_filename)
        count = 0
        success = 1

        errored = False

        while success:
            success, image = vidObj.read()
            if success != 1:
                break

            if count % 5 == 0:
                try:
                    cv2.imwrite(image_filename, image)
                    img = dlib.load_grayscale_image(image_filename)
                    appears_drowsy = detector.areEyesClosed(img)

                    print(appears_drowsy)
                    print("current consecutive drowsy frames: ", detector.getNumberConsecutiveDrowsyFrames())
                    if appears_drowsy is not None:
                        if not appears_drowsy:
                            detector.resetNumberConsecutiveDrowsyFrames()

                        if detector.isDrowsy():
                            response["drowsy"] = True
                            response["image"] = image_filename.replace("\\", "/")
                            response["video"] = video_filename.replace("\\", "/")
                            break

                except Exception as e:
                    errored = True
                    print(e)
                    break

            count += 1
        print(count)

        if errored:
            return jsonify(response), 500
        else:
            return jsonify(response), 200

    except Exception as e:
        print(e)
        return jsonify({"error": "Invalid request"}), 400

@app.route('/storage/video/<filename>')
def serve_video(filename):
    video_path = os.path.join(VIDEO_DIR, filename)
    return send_file(video_path, mimetype='video/mp4')

@app.route('/storage/image/<filename>')
def serve_image(filename):
    image_path = os.path.join(IMAGE_DIR, filename)
    return send_file(image_path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0',port=8001)
