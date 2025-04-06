from PIL import Image
from ultralytics import YOLO

model = YOLO('runs/train1V10/weights/best.pt')  # load a pretrained model (recommended for training)

# Set confidence threshold (e.g., 0.5)
confidence_threshold = 0.2

# Run batched inference on a list of images with the confidence threshold
results = model(['/home/dxcheng/pothole_detection_yolo_v10/2024-07-30_16-25-21pothole.jpg'], conf=confidence_threshold)  # return a list of Results objects

# # Run batched inference on a list of images
# results = model(['/home/dxcheng/road_vision/stitching/tmp/outdir/panorama_9.jpg'])  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs

# Show the results
for r in results:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image
    im.save('results.jpg')  # save image