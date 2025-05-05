from ultralytics import YOLO
import os


# model1 = YOLO('yolo11n.pt')  #Object Detection model
model1 = YOLO('yolo11n-seg.pt')

save_dir = 'runs'
os.makedirs(save_dir, exist_ok=True)    

# Test final learning rates with Stochastic Gradient Descent
results = model1.train(name='train' , data='data/v1/data.yaml', epochs=80, imgsz=640, device=[0,1,2,3,4,5,6,7], project=save_dir, optimizer='SGD')

#results = model1.train(name='AdamW_lrf1',data='v1/data.yaml', epochs=100, imgsz=640, device=[0], project=save_dir, optimizer='AdamW')

#results = model4.train(name='AdamW_lrf1'   , data='v5/data.yaml', epochs=300, imgsz=640, device=[0], optimizer='AdamW', lr0=0.03, lrf=1)
#results = model5.train(name='AdamW_lrf0.5' , data='v5/data.yaml', epochs=300, imgsz=640, device=[0], optimizer='AdamW', lr0=0.03, lrf=0.5)
#results = model6.train(name='AdamW_lrf0.03', data='v5/data.yaml', epochs=300, imgsz=640, device=[0], optimizer='AdamW', lr0=0.03, lrf=0.04)

# # Baseline models for comparison
# results = model7.train(name='SGD_lri0.01'  , data='v5/data.yaml', epochs=900, imgsz=640, device=[0], optimizer='SGD'  , lr0=0.01, lrf=1, patience = 100)
# results = model8.train(name='AdamW_lri0.01', data='v5/data.yaml', epochs=900, imgsz=640, device=[0], optimizer='AdamW', lr0=0.01, lrf=1, patience = 100)

print("Training Complete")