from ultralytics.models.yolo.model import YOLO

def train_yolo():
    # 1. Load model configuration
    # model = YOLO("ultralytics/cfg/models/11/yolo11s.yaml")  # load model structure from current directory
    model = YOLO("yolov11s-FlexInc.yaml")

    # 2. Manually load pretrained weights
    # model.load(r"yolo11s.pt")  # ensure weights file exists in current directory

    # 3. Configure training parameters
    results = model.train(
        data=r"CellDetectSampleDataset/CellDetectSampleDataset.yaml",  # dataset config file
        # imgsz=1280,  # input image size
        imgsz=640,  # input image size
        # scale=0.5,  # multi-scale range: 640×0.5=320 to 640×1.5=960
        epochs=100,  # training epochs
        batch=8,  # batch size (adjust based on GPU memory)
        workers=4,  # data loading threads
        pretrained=False,  # pretrained weights already loaded
        device='0',  # use GPU 0 (empty string for auto-selection)
        seed=0,  # random seed for reproducibility
        optimizer='Adam',  # optimizer choice
        amp=False,
        project='FlexInc-YOLO',  # project name for saving results
        resume=False,  # do not resume training
        name='FlexInc-YOLO_CellDetectSampleDataset'  # training task name
    )
    return results
    
if __name__ == "__main__":
    train_yolo()