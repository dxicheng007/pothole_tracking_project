from ultralytics import YOLO
from datetime import datetime
import logging
import time
import os


# Configure logging
log_file = 'training_log.txt'
logging.basicConfig(filename=log_file, level=logging.INFO, 
                    format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Log start time
start_time = time.time()
start_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
logging.info(f"Training started at: {start_datetime}")

# Initialize model
pretrained_model = 'yolo11n-seg.pt'
model1 = YOLO(pretrained_model)
logging.info(f"Model initialized with pretrained weights: {pretrained_model}")

# Set up save directory
save_dir = 'runs'
os.makedirs(save_dir, exist_ok=True)    
logging.info(f"Base results directory: {os.path.abspath(save_dir)}")

# Log data configuration
data_config = 'data/v1/data.yaml'
logging.info(f"Using data config: {data_config}")

# Start training
try:
    results = model1.train(
        name='train',  # Ultralytics will auto-increment (train, train2, train3, etc.)
        data=data_config,
        epochs=80,
        imgsz=640,
        batch=-1,         # <--- find biggest batch automatically!
        device=[0,1],
        project=save_dir
    )
    
    # Get the exact training run folder (e.g., runs/train8)
    # Ultralytics saves the last run path in results.save_dir
    training_run_dir = results.save_dir if hasattr(results, 'save_dir') else "Unknown"
    logging.info(f"Exact training folder: {training_run_dir}")
    
    # Log completion
    end_time = time.time()
    end_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    training_duration = (end_time - start_time) / 3600  # in hours
    
    logging.info(f"Training completed at: {end_datetime}")
    logging.info(f"Total training time: {training_duration:.2f} hours")
    logging.info(f"Full results path: {training_run_dir}")
    
    print(f"Training completed. Results saved in: {training_run_dir}")
    print(f"Log file saved as: {log_file}")

except Exception as e:
    logging.error(f"Training failed with error: {str(e)}")
    print(f"Training failed. Error logged in {log_file}")
    raise
print("Training Complete")