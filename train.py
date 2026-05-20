from ultralytics import YOLO


if __name__ == "__main__":

    model = YOLO(r".\TLK-YOLO.yaml")

    model.info()

    # Train the model on the COCO8 dataset for 100 epochs
    # # #
    train_results = model.train(
         data=r".\datasets\PDT.yaml",  # Path to dataset configuration file
         epochs=300,  # Number of training epochs
         imgsz=640,  # Image size for training
         device="0",  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
         patience=0,
         cache=False,
         batch=8,
         optimizer="SGD",
         lr0=0.01,
         lrf=0.1,
         # resume=True
         save_json=False,
         project='PDT',
     )


