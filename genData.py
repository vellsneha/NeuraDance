import mediapipe as mp
import numpy as np
import cv2
import os



# Paths
video_folder = "C:\\Users\\sneha\\Desktop\\Academics\\NewProjects\\CV_Dance\\DanceForms"
output_folder = "C:\\Users\\sneha\\Desktop\\Academics\\NewProjects\\CV_Dance\\DataGen"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Initialize MediaPipe Pose
mp_pose = mp.sSolutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
drawing = mp.solutions.drawing_utils



# Loop through all video files in the folder
for video_file in os.listdir(video_folder):
    if not video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        continue  # Skip non-video files

    video_path = os.path.join(video_folder, video_file)
    cap = cv2.VideoCapture(video_path)

    # Check if the video file is loaded correctly
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()

    print(f"\nProcessing video: {video_file}")
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

# Save data
output_name = os.path.splitext(video_file)[0]  # Remove file extension
save_path = os.path.join(output_folder, f"{output_name}.npy")
np.save(save_path, np.array(X))
print(f"Saved: {save_path} | Shape: {np.array(X).shape} | Frames: {data_size}")
