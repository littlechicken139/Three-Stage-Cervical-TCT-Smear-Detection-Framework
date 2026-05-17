from ultralytics.models.yolo.model import YOLO

# Load model
model = YOLO("runs\detect\FlexInc-YOLO\FlexInc-YOLO_CellDetectSampleDataset\weights\best.pt")  # or custom trained model

# Run prediction and save results
results = model.predict(
    conf=0.25,
    source=r"CellDetectSampleDataset\test",  # image directory or single image path
    save_txt=True,          # save txt annotation files
    save=True,              # save images with bounding boxes
    project="precict_model",  # custom save root directory
    name="FlexInc-YOLO_testset",  # custom subdirectory name    
    # save_conf=True          # include confidence in txt (default: False)
    # visualize=True          # visualize model feature maps
    # line_width=2            # bounding box line width
    show_conf=True,         # show prediction confidence
    show_labels=True,       # show prediction labels
    # save_crop=True          # save cropped detection images
)

# Verify save path
print(f"Prediction results saved to: {results[0].save_dir}")