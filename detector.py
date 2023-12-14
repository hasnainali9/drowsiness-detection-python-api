import dlib
import cv2
import numpy
from imutils import face_utils
from scipy.spatial import distance as dist
import os

"""
A class to represent a machine capable of determing whether or not a
person appears drowsy based on an ordered collection of images provided in
short bursts.

Attributes
----------
_consecutiveDrowsyFrames: An integer representing the number of consecutive
    frames in which the subject has thier eyes closed.
_maxDrowsyFramesBeforeSignal: An integer representing the number of consecutive
    frames in which the subject must have his/her eyes closed before a call to
    detect() will return True, indicating that the driver appears drowsy.
_minimumEyeAspectRatioBeforeCloseAssumed: A floating point value representing
    the minimum ratio between the height and width of the subject's eyes before
    it is decided that the user's eyes are closed.

Methods
-------
areEyesClosed(images)
    Returns True if, after analyzing images, it is determined that the person
    depicted in the images in drowsy, False otherwise.
getEyeAspectRatio(eye)
    Returns eye aspect ratio given ndarry containing coordinates of eyes.
"""
class DrowsinessDetector:
    def __init__(self):
        # dotenv.load_dotenv()
        self._consecutiveDrowsyFrames = 0
        self._maxDrowsyFramesBeforeSignal =  3
            # int(os.getenv("FRAMES_BEFORE_DROWSINESS_CONFIRMED"))
        self._minimumEyeAspectRatioBeforeCloseAssumed = 0.25
            # float(os.getenv("MINIMUM_EYE_ASPECT_RATIO_BEFORE_ASSUMED_CLOSED"))

    """
    Analyzes eyes appearing in image and makes a determination based on
    the number of frames for which the face in the images has had his/her
    eyes closed as to whether the images depict a drowsy person and makes
    a determination one way or the other.

    @param img List[List[int]] A 2-dimensional ndarray representing a
        an image including only a single face.
    @return bool True if the enough frames have passed that the person depicted
        in the images can be considered drowsy, else False.
    """
    def areEyesClosed(self, img):
        landmarkDetector = dlib.get_frontal_face_detector()
        shapePredictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        dets = landmarkDetector(img, 1)
        print("num faces: ", len(dets))

        if not dets:
            return False

        facialLandmarks = shapePredictor(img, dets[0])
        # grab the indexes of the facial landmarks for the left and
        # right eye, respectively
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        shape = face_utils.shape_to_np(facialLandmarks)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = self.getEyeAspectRatio(leftEye)
        rightEAR = self.getEyeAspectRatio(rightEye)

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(img, [rightEyeHull], -1, (0, 255, 0), 1)
        ear = (leftEAR + rightEAR) / 2
        headmovement=detect_head_movement(img);
        print(ear)

        if ear < self._getMinimumEyeAspectRatio():
            self.incrementNumberConsecutiveDrowsyFrames()
            return True

        return False


    def getEyeAspectRatio(self, eye):
        # compute the euclidean distances between the two sets of
        # vertical eye landmarks (x, y)-coordinates
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])

        # compute the euclidean distance between the horizontal
        # eye landmark (x, y)-coordinates
        C = dist.euclidean(eye[0], eye[3])

        ear = (A + B) / (2.0 * C)

        return ear

    def isDrowsy(self):
        return self.getNumberConsecutiveDrowsyFrames() > \
               self.getMaxDrowsyFramesBeforeSignal()

    def _getMinimumEyeAspectRatio(self):
        return self._minimumEyeAspectRatioBeforeCloseAssumed

    def getMaxDrowsyFramesBeforeSignal(self):
        return self._maxDrowsyFramesBeforeSignal

    def getNumberConsecutiveDrowsyFrames(self):
        return self._consecutiveDrowsyFrames

    def incrementNumberConsecutiveDrowsyFrames(self):
        self._consecutiveDrowsyFrames += 1

    def resetNumberConsecutiveDrowsyFrames(self):
        self._consecutiveDrowsyFrames = 0

    def detect_head_movement(shape):
        # Implement your head movement detection logic here
        # Return a value indicating the extent of head movement
        return 0
if __name__ == "__main__":
    d = DrowsinessDetector()
    img = dlib.load_grayscale_image("testImage.png")
    print(d.areEyesClosed(img))




