import cv2
import os

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

video_path = "data/videos/GX011362potholes.MP4"

model = YOLO("runs/train3/weights/best.pt")  # segmentation model
cap = cv2.VideoCapture(video_path)


w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Generate the new video path
base, ext = os.path.splitext(video_path)
new_video_path = f"{base}_seg-tracking_train3{ext}"

# # Alternative way to generate the new video path
# new_video_path = video_path.replace(".MP4", "_seg-tracking.MP4")
out = cv2.VideoWriter(new_video_path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

# Define the desired window size
window_width = 1280  # Example width
window_height = 720  # Example height

# Create a named window with the specified size
cv2.namedWindow("instance-segmentation-object-tracking", cv2.WINDOW_NORMAL)
cv2.resizeWindow("instance-segmentation-object-tracking", window_width, window_height)

num_potholes=0
pot_holes_tag=0
pot_holes_id=[]
while True:
    ret, im0 = cap.read()
    if not ret:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    annotator = Annotator(im0, line_width=5)

    results = model.track(im0, persist=True)
    
    if results[0].boxes.id is not None and results[0].masks is not None:
        masks = results[0].masks.xy
        track_ids = results[0].boxes.id.int().cpu().tolist()
        

        for mask, track_id in zip(masks, track_ids):
            color = colors(int(track_id), True)
            txt_color = annotator.get_txt_color(color)
            annotator.seg_bbox(mask=mask, mask_color=color, label=str(track_id), txt_color=txt_color)
            print("Track ID: ", track_id)
            if track_id!=pot_holes_tag:
                num_potholes+=1
                pot_holes_tag=track_id
                pot_holes_id.append(track_id)

    out.write(im0)
    cv2.imshow("instance-segmentation-object-tracking", im0)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

out.release()
cap.release()
cv2.destroyAllWindows()
print("Potholes ID: ", pot_holes_id)
print("Number of potholes detected: ", num_potholes)