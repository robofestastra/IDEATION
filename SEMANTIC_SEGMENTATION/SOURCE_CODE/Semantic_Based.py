import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PERSON_CLASS_ID = 15


def load_model():
    try:
        import torch
        from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
    except ImportError as e:
        sys.exit(f"PyTorch/torchvision is not installed.\nRun:  pip install torch torchvision opencv-python\nOriginal error: {e}")

    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    model = deeplabv3_resnet50(weights=weights)
    model.eval()
    preprocess = weights.transforms()
    return model, preprocess


def collect_images(input_dir: Path):
    images = sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        sys.exit(f"No images found in {input_dir} (looked for {IMAGE_EXTS})")
    return images


def run_segmentation(model, preprocess, image_paths, output_dir: Path):
    import cv2
    import torch
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    for idx, img_path in enumerate(image_paths, start=1):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"[skip] could not read image: {img_path}")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        input_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        input_batch = preprocess(input_tensor).unsqueeze(0)

        t0 = time.time()
        with torch.no_grad():
            output = model(input_batch)["out"][0]
        inference_ms = (time.time() - t0) * 1000

        pred_mask = output.argmax(0).byte().cpu().numpy()
        pred_mask_resized = cv2.resize(pred_mask.astype(np.uint8), (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

        person_mask = pred_mask_resized == PERSON_CLASS_ID
        person_pixel_count = int(person_mask.sum())
        total_pixels = pred_mask_resized.shape[0] * pred_mask_resized.shape[1]
        person_pixel_pct = round(100 * person_pixel_count / total_pixels, 2)
        victim_present = person_pixel_count > 0

        overlay = img_bgr.copy()
        colored_mask = np.zeros_like(img_bgr, dtype=np.uint8)
        colored_mask[person_mask] = (0, 0, 255)
        overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.45, 0)

        if victim_present:
            ys, xs = np.where(person_mask)
            label_x, label_y = int(xs.min()), max(0, int(ys.min()) - 8)
            cv2.putText(overlay, f"VICTIM ({person_pixel_pct}% of frame)", (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)

        out_path = output_dir / f"semantic_{img_path.stem}.jpg"
        cv2.imwrite(str(out_path), overlay)
        print(f"[{idx}/{len(image_paths)}] {img_path.name}: person-class coverage {person_pixel_pct}% of frame, victim_present={victim_present}, {inference_ms:.1f} ms -> saved {out_path.name}")

        all_results.append({
            "source_image": str(img_path.name),
            "annotated_image": str(out_path.name),
            "task": "semantic_segmentation",
            "inference_ms": round(inference_ms, 2),
            "person_pixel_count": person_pixel_count,
            "person_pixel_pct_of_frame": person_pixel_pct,
            "victim_present": victim_present,
        })

    return all_results


def write_log(all_results, output_dir: Path):
    log_path = output_dir / "detections.json"
    with open(log_path, "w") as f:
        json.dump(all_results, f, indent=2)

    summary_path = output_dir / "summary.txt"
    total_images = len(all_results)
    total_victim_frames = sum(r["victim_present"] for r in all_results)
    avg_inference = sum(r["inference_ms"] for r in all_results) / total_images if total_images else 0
    avg_coverage = sum(r["person_pixel_pct_of_frame"] for r in all_results) / total_images if total_images else 0

    with open(summary_path, "w") as f:
        f.write("ASTRA - Semantic Segmentation Simulation Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Images processed          : {total_images}\n")
        f.write(f"Frames with victim pixels : {total_victim_frames}\n")
        f.write(f"Avg person-class coverage : {avg_coverage:.2f}% of frame\n")
        f.write(f"Avg inference time (ms)   : {avg_inference:.2f}\n")

    print(f"\nDetection log  -> {log_path}")
    print(f"Run summary    -> {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="ASTRA RGB-channel semantic segmentation simulation (DeepLabV3-ResNet50).")
    parser.add_argument("--input", default="./Victim", help="Folder containing locally stored test images (default: ./Victim)")
    parser.add_argument("--output", default="./semantic_segmentations", help="Folder to write annotated images + detection log (default: ./semantic_segmentations)")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    if not input_dir.exists():
        sys.exit(f"Input folder does not exist: {input_dir}")

    print("Loading model: DeepLabV3-ResNet50 (torchvision, pretrained)")
    model, preprocess = load_model()
    images = collect_images(input_dir)
    print(f"Found {len(images)} image(s) in {input_dir}\n")

    results = run_segmentation(model, preprocess, images, output_dir)
    write_log(results, output_dir)


if __name__ == "__main__":
    main()
