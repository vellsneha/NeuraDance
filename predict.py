import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model

# Function to check if full body is visible
def inFrame(landmarks):
    if (
        landmarks[28].visibility > 0.6 and
        landmarks[27].visibility > 0.6 and
        landmarks[15].visibility > 0.6 and
        landmarks[16].visibility > 0.6
    ):
        return True
    return False

# Load model and labels
model = load_model("model.h5")
label = np.load("labels.npy")

# Initialize MediaPipe Pose
holistic = mp.solutions.pose
pose = holistic.Pose()

def predict_pose(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return

    # Convert to RGB and process with MediaPipe Pose
    res = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # if res.pose_landmarks and inFrame(res.pose_landmarks.landmark):
    ''' 
    removing the condition for the whole body to be in the frame
    Issues:
    (1) not able to detect half body poses, such as image B2.jpg
    '''
    if res.pose_landmarks:
        keypoints = []
        for i in res.pose_landmarks.landmark:
            keypoints.append(i.x - res.pose_landmarks.landmark[0].x)
            keypoints.append(i.y - res.pose_landmarks.landmark[0].y)

        keypoints = np.array(keypoints).reshape(1, -1)

        # Predict pose
        prediction = model.predict(keypoints)
        predicted_label = label[np.argmax(prediction)]

        # Print result in command line
        if prediction[0][np.argmax(prediction)] > 0.75:
            print(f"Predicted Pose: {predicted_label}")
        else:
            print("Pose is either incorrect or not trained.")
    else:
        print("Make sure the full body is visible in the image.")

if __name__ == "__main__":
    image_path = "C:\\Users\\sneha\\Desktop\\Academics\\NewProjects\\CV_Dance\\ImagesToPredict\\K5.jpg"
    if len(image_path) < 2:
        print(f"Usage: python script.py {image_path}")
    else:
        predict_pose(image_path)
