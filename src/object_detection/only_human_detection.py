import cv2
import yaml
from ultralytics import YOLO

# custom tracker system
custom_tracker = {
    'tracker_type': 'botsort',
    'track_high_thresh': 0.5,
    'track_low_thresh': 0.1,
    'new_track_thresh': 0.6,
    'track_buffer': 300,          # memory(approximately 10 second)
    'match_thresh': 0.8,
    'gmc_method': 'sparseOptFlow',
    'proximity_thresh': 0.5,
    'appearance_thresh': 0.25,
    'with_reid': True,            # Re-ID 
    'model': 'yolo11n-cls.pt',    
    'fuse_score': True
}

# save the configuration
with open("custom_reid_tracker.yaml", "w") as f:
    yaml.dump(custom_tracker, f)

def run_ultralytics_reid_system():

    model = YOLO("yolo11s.pt")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ক্যামেরা ওপেন করা যাচ্ছে না! অন্য কোথাও ব্যবহার হচ্ছে কি না চেক করো।")
        return

    unique_human_ids = set()

    print("\nUltralytics Re-ID সিস্টেম চালু হয়েছে... (বন্ধ করতে 'q' চাপুন)\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ফ্রেম রিড করা যাচ্ছে না!")
            break

        
        results = model.track(frame, persist=True, classes=[0], tracker="custom_reid_tracker.yaml", conf=0.6, verbose=False)

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, track_id in zip(boxes, ids):
                unique_human_ids.add(track_id)
                
                # Draw the box and give id
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id}", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # show the count in live on screen
        cv2.putText(frame, f"Total Unique Count: {len(unique_human_ids)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.imshow("Ultralytics Built-in Re-ID", frame)
        
        # 'q' button for exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# run the code
run_ultralytics_reid_system()