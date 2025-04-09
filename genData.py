import mediapipe as mp
import numpy as np
import cv2



# Load the video
video_path = "C:\\Users\\sneha\\Desktop\\Academics\\NewProjects\\CV_Dance\\MohiniyatamPerformance.mp4"
cap = cv2.VideoCapture(video_path)


# Check if the video file is loaded correctly
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()


# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
drawing = mp.solutions.drawing_utils

# Initialize data storage
X = []
data_size = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break  # Exit if the video ends
    
    frame = cv2.flip(frame, 1)  # Flip for a mirrored view
    res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    if res.pose_landmarks:
        print("Pose detected!")
        lst = []
        for landmark in res.pose_landmarks.landmark:
            lst.append(landmark.x - res.pose_landmarks.landmark[0].x)  # Relative X
            lst.append(landmark.y - res.pose_landmarks.landmark[0].y)  # Relative Y
        X.append(lst)
        data_size += 1
    else:
        print("No pose detected in this frame.")
    
    # Draw pose landmarks
    drawing.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    # Show frame count
    cv2.putText(frame, f"Frames Captured: {data_size}", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Pose Detection", frame)
    if cv2.waitKey(1) == 27:  # Press 'Esc' to exit early
        break

cap.release()
cv2.destroyAllWindows()

# Save the extracted data
output_name = "Performance"
np.save(f"{output_name}.npy", np.array(X))
print(f"Data saved: {output_name}.npy with shape {np.array(X).shape}")
print(f"Total frames processed: {data_size}")

