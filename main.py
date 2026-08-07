import cv2
import numpy as np
import torch
from ultralytics import YOLO

def analyze_dtl_swing(video_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Running YOLO Pose on device: {device}")

    # Load YOLOv8 Pose model (downloads automatically on first run ~6MB)
    model = YOLO("yolov8n-pose.pt").to(device)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video file at: {video_path}")
        return

    wrist_trajectory = []
    initial_hip_x = None

    # COCO Keypoint Index Mapping:
    # 5: Left Shoulder, 6: Right Shoulder
    # 9: Left Wrist, 10: Right Wrist
    # 11: Left Hip, 12: Right Hip

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape

        # Run YOLO pose inference on GPU
        results = model(frame, verbose=False)[0]

        # Extract detected pose keypoints
        if results.keypoints is not None and len(results.keypoints.data) > 0:
            # First detected person in frame (x, y, confidence)
            kpts = results.keypoints.data[0].cpu().numpy()

            # Right Wrist (Keypoint 10) & Right Hip (Keypoint 12)
            r_wrist = (int(kpts[10][0]), int(kpts[10][1]))
            r_hip = (int(kpts[12][0]), int(kpts[12][1]))
            r_shoulder = (int(kpts[6][0]), int(kpts[6][1]))

            # Only track if keypoints are visible (confidence > 0.3)
            if kpts[10][2] > 0.3:
                wrist_trajectory.append(r_wrist)

            # Establish baseline hip line (Tush Line) at address frame
            if initial_hip_x is None and kpts[12][2] > 0.3:
                initial_hip_x = r_hip[0]

            # 1. Draw Early Extension Reference Line
            if initial_hip_x is not None:
                cv2.line(frame, (initial_hip_x, 0), (initial_hip_x, h), (0, 255, 255), 2)

                # Calculate horizontal pelvic drift
                hip_drift = r_hip[0] - initial_hip_x
                cv2.putText(frame, f"Hip Drift: {hip_drift} px", (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            # 2. Draw Hand Trajectory Trail (Backswing & Downswing path)
            for i in range(1, len(wrist_trajectory)):
                cv2.line(frame, wrist_trajectory[i - 1], wrist_trajectory[i], (0, 0, 255), 2)

            # 3. Calculate Spine Angle (Shoulder to Hip vector)
            if kpts[6][2] > 0.3 and kpts[12][2] > 0.3:
                dx = r_shoulder[0] - r_hip[0]
                dy = r_shoulder[1] - r_hip[1]
                spine_angle = np.degrees(np.arctan2(-dy, dx))
                cv2.putText(frame, f"Spine Angle: {int(spine_angle)} deg", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Draw keypoint dots on body
            cv2.circle(frame, r_wrist, 5, (0, 0, 255), -1)
            cv2.circle(frame, r_hip, 5, (255, 0, 0), -1)

        cv2.imshow("DTL Golf Analyzer (YOLO Pose)", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Note: Make sure swing.mp4 is in the same directory, or pass the exact path
    analyze_dtl_swing("swing.mp4")