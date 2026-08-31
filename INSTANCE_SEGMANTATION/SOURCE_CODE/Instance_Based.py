import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

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


def run_segmentation(model, image_paths, conf_threshold: float, output_dir: Path):
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
        overlay = img.copy()
        has_masks = r.masks is not None

        for i, box in enumerate(r.boxes):
            cls_id = int(box.cls.item())
            cls_name = model.names[cls_id]
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            is_victim_proxy = cls_name in VICTIM_PROXY_CLASSES
            color = (0, 0, 255) if is_victim_proxy else (0, 180, 180)
            label = f"{cls_name} {confidence:.2f}"
            if is_victim_proxy:
                label = f"VICTIM ({label})"

            mask_pixel_count = 0
            if has_masks:
                mask = r.masks.data[i].cpu().numpy()
                mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask_bool = mask_resized.astype(bool)
                mask_pixel_count = int(mask_bool.sum())
                colored_mask = np.zeros_like(img, dtype=np.uint8)
                colored_mask[mask_bool] = color
                overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.45, 0)

            cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(overlay, label, (int(x1), max(0, int(y1) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 5)

            detections.append({
                "class": cls_name,
                "confidence": round(confidence, 4),
                "bbox_xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "mask_pixel_count": mask_pixel_count,
                "victim_candidate": is_victim_proxy,
            })

        out_path = output_dir / f"segmented_{img_path.stem}.jpg"
        cv2.imwrite(str(out_path), overlay)
        victim_count = sum(d["victim_candidate"] for d in detections)
        print(f"[{idx}/{len(image_paths)}] {img_path.name}: {len(detections)} instances, {victim_count} victim(s), {inference_ms:.1f} ms -> saved {out_path.name}")

        all_results.append({
            "source_image": str(img_path.name),
            "annotated_image": str(out_path.name),
            "task": "instance_segmentation",
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
    total_instances = sum(len(r["detections"]) for r in all_results)
    total_victims = sum(r["victim_count"] for r in all_results)
    avg_inference = sum(r["inference_ms"] for r in all_results) / total_images if total_images else 0

    with open(summary_path, "w") as f:
        f.write("ASTRA - Instance Segmentation Simulation Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Images processed        : {total_images}\n")
        f.write(f"Total instances         : {total_instances}\n")
        f.write(f"Victims                 : {total_victims}\n")
        f.write(f"Avg inference time (ms) : {avg_inference:.2f}\n")

    print(f"\nDetection log  -> {log_path}")
    print(f"Run summary    -> {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="ASTRA RGB-channel instance segmentation simulation (YOLOv8n-seg).")
    parser.add_argument("--input", default="./Victim", help="Folder containing locally stored test images (default: ./Victim)")
    parser.add_argument("--output", default="./segmentations", help="Folder to write annotated images + detection log (default: ./segmentations)")
    parser.add_argument("--model", default="yolov8n-seg.pt", help="Ultralytics segmentation weights (default: yolov8n-seg.pt, auto-downloads on first run)")
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

    results = run_segmentation(model, images, args.conf, output_dir)
    write_log(results, output_dir)


if __name__ == "__main__":
    main()
