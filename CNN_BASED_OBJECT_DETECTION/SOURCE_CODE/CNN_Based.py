import argparse
import json
import sys
import time
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VICTIM_PROXY_CLASSES = {"person"}


def load_model(weights: str):
    try:
        from ultralytics import YOLO
    except ImportError as e:
        sys.exit(f"Ultralytics is not installed.\nRun:  pip install ultralytics opencv-python\nOriginal error: {e}")
    return YOLO(weights)


def collect_images(input_dir: Path):
    images = sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        sys.exit(f"No images found in {input_dir} (looked for {IMAGE_EXTS})")
    return images


def run_detection(model, image_paths, conf_threshold: float, output_dir: Path):
    import cv2
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    for idx, img_path in enumerate(image_paths, start=1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[skip] could not read image: {img_path}")
            continue

        t0 = time.time()
        results = model.predict(source=img, conf=conf_threshold, verbose=False)
        inference_ms = (time.time() - t0) * 1000
        r = results[0]
        detections = []

        for box in r.boxes:
            cls_id = int(box.cls.item())
            cls_name = model.names[cls_id]
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            is_victim_proxy = cls_name in VICTIM_PROXY_CLASSES
            color = (0, 0, 255) if is_victim_proxy else (0, 200, 0)
            label = f"{cls_name} {confidence:.2f}"
            if is_victim_proxy:
                label = f"VICTIM ({label})"

            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(img, label, (int(x1), max(0, int(y1) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 5)

            detections.append({
                "class": cls_name,
                "confidence": round(confidence, 4),
                "bbox_xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "victim": is_victim_proxy,
            })

        out_path = output_dir / f"annotated_{img_path.stem}.jpg"
        cv2.imwrite(str(out_path), img)
        victim_count = sum(d["victim"] for d in detections)
        print(f"[{idx}/{len(image_paths)}] {img_path.name}: {len(detections)} detections, {victim_count} victim(s), {inference_ms:.1f} ms -> saved {out_path.name}")

        all_results.append({
            "source_image": str(img_path.name),
            "annotated_image": str(out_path.name),
            "inference_ms": round(inference_ms, 2),
            "detections": detections,
            "victim_count": victim_count,
            "victim_confirmed": victim_count > 0,
        })

    return all_results


def write_log(all_results, output_dir: Path):
    log_path = output_dir / "detections.json"
    with open(log_path, "w") as f:
        json.dump(all_results, f, indent=2)

    summary_path = output_dir / "summary.txt"
    total_images = len(all_results)
    total_detections = sum(len(r["detections"]) for r in all_results)
    total_victims = sum(r["victim_count"] for r in all_results)
    avg_inference = sum(r["inference_ms"] for r in all_results) / total_images if total_images else 0

    with open(summary_path, "w") as f:
        f.write("ASTRA - CNN Object Detection Simulation Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Images processed        : {total_images}\n")
        f.write(f"Total detections        : {total_detections}\n")
        f.write(f"Victim boxes  : {total_victims}\n")
        f.write(f"Avg inference time (ms) : {avg_inference:.2f}\n")

    print(f"\nDetection log  -> {log_path}")
    print(f"Run summary    -> {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="ASTRA RGB-channel CNN object detection simulation (YOLOv8n).")
    parser.add_argument("--input", required=True, help="Folder containing locally stored test images")
    parser.add_argument("--output", default="./detections", help="Folder to write annotated images + detection log (default: ./detections)")
    parser.add_argument("--model", default="yolov8n.pt", help="Ultralytics weights file/name (default: yolov8n.pt, auto-downloads on first run)")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold for a detection to be kept (default: 0.4)")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    if not input_dir.exists():
        sys.exit(f"Input folder does not exist: {input_dir}")

    print(f"Loading model: {args.model}")
    model = load_model(args.model)
    images = collect_images(input_dir)
    print(f"Found {len(images)} image(s) in {input_dir}\n")

    results = run_detection(model, images, args.conf, output_dir)
    write_log(results, output_dir)


if __name__ == "__main__":
    main()
