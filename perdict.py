from ultralytics import YOLO


if __name__ == "__main__":

    model = YOLO('./best.pt')
    # model.info()
    model.info()

    metrics = model.predict(
        source=r".\images",
        save=True,
        project='PDT',
        line_width=2,
        show_labels=False,
        conf=0.3,
        # iou=0.7,
    )


