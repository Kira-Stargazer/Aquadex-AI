from ultralytics import YOLO


MODEL_PATH = "best.pt"


def run_inference(
    image_path,
    confidence=0.5,
    image_size=1024
):
    model = YOLO(MODEL_PATH)

    results = model.predict(
        source=image_path,
        conf=confidence,
        imgsz=image_size,
        verbose=False
    )

    return results


if __name__ == "__main__":

    image_path = "sample_input.png"

    results = run_inference(
        image_path=image_path,
        confidence=0.5,
        image_size=1024
    )

    print("Inference completed.")
