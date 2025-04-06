from ultralytics import YOLO
import os

save_dir = 'runs'
os.makedirs(save_dir, exist_ok=True)  
# Evaluate the model's performance on the validation set
model = YOLO("runs/train/weights/best.pt")
# results = model.val(data='data/v1/data.yaml', project=save_dir)

## Show the result for one image:
# Perform prediction on an image using the model
results = model("data/v1/test/images/frame_381_jpg.rf.436029bcc12c84452a44bb472e401e51.jpg")
# Visualize the predictions
result_image = results[0].plot()
# Display the image using an appropriate library, e.g., OpenCV or Matplotlib
import matplotlib.pyplot as plt
plt.imshow(result_image)
plt.axis('on')
plt.show()


