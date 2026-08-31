import cv2
import numpy as np
import os

IMAGE_PATH = "Thermal_2.webp"
OUTPUT_PATH = "thermal_hotspot_2.jpg"

HAS_TEMPERATURE_DATA = False
MIN_TEMP = 20.0
MAX_TEMP = 80.0

HOTSPOT_THRESHOLD = 85
MIN_BLOB_AREA = 50
MAX_BLOB_AREA = None
BLUR_KERNEL = 5
MORPH_KERNEL_SIZE = 5
MIN_CONFIDENCE = 50.0

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(
        f"Input image not found:\n{IMAGE_PATH}\n\n"
        "Place your thermal image in the same folder as this script "
        "or change IMAGE_PATH."
    )

image = cv2.imread(IMAGE_PATH)
if image is None:
    raise ValueError("Unable to read the thermal image.")

original = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_blur = cv2.GaussianBlur(gray, (BLUR_KERNEL, BLUR_KERNEL), 0)
normalized = cv2.normalize(gray_blur, None, 0, 255, cv2.NORM_MINMAX)

threshold_value = int((HOTSPOT_THRESHOLD / 100.0) * 255)
_, hotspot_mask = cv2.threshold(normalized, threshold_value, 255, cv2.THRESH_BINARY)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
hotspot_mask = cv2.morphologyEx(hotspot_mask, cv2.MORPH_OPEN, kernel)
hotspot_mask = cv2.morphologyEx(hotspot_mask, cv2.MORPH_CLOSE, kernel)

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(hotspot_mask, connectivity=8)

detected_hotspots = []

for i in range(1, num_labels):
    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    area = stats[i, cv2.CC_STAT_AREA]
    cx, cy = centroids[i]

    if area < MIN_BLOB_AREA:
        continue
    if MAX_BLOB_AREA is not None and area > MAX_BLOB_AREA:
        continue

    component_mask = (labels == i)
    pixel_values = normalized[component_mask]
    mean_intensity = float(np.mean(pixel_values))
    max_intensity = float(np.max(pixel_values))

    if HAS_TEMPERATURE_DATA:
        temperature = MIN_TEMP + (mean_intensity / 255.0) * (MAX_TEMP - MIN_TEMP)
        max_temperature = MIN_TEMP + (max_intensity / 255.0) * (MAX_TEMP - MIN_TEMP)
    else:
        temperature = None
        max_temperature = None

    intensity_score = ((mean_intensity - threshold_value) / (255 - threshold_value)) * 100
    intensity_score = np.clip(intensity_score, 0, 100)

    area_score = min((area / (MIN_BLOB_AREA * 10)) * 100, 100)

    perimeter_mask = component_mask.astype(np.uint8)
    contours, _ = cv2.findContours(perimeter_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        perimeter = cv2.arcLength(contours[0], True)
        if perimeter > 0:
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            shape_score = np.clip(circularity * 100, 0, 100)
        else:
            shape_score = 0
    else:
        shape_score = 0

    confidence = 0.50 * intensity_score + 0.30 * area_score + 0.20 * shape_score

    hotspot = {
        "id": len(detected_hotspots) + 1,
        "x": x, "y": y, "width": w, "height": h, "area": area,
        "centroid": (cx, cy),
        "mean_intensity": mean_intensity,
        "max_intensity": max_intensity,
        "confidence": confidence,
        "temperature": temperature,
        "max_temperature": max_temperature,
    }

    if confidence >= MIN_CONFIDENCE:
        detected_hotspots.append(hotspot)

result = original.copy()

for hotspot in detected_hotspots:
    x, y, w, h = hotspot["x"], hotspot["y"], hotspot["width"], hotspot["height"]
    cx, cy = hotspot["centroid"]
    confidence = hotspot["confidence"]

    cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.circle(result, (int(cx), int(cy)), 5, (255, 0, 0), -1)
    label = f"Hotspot {hotspot['id']} | {confidence:.1f}%"
    cv2.putText(result, label, (x, max(y - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)

cv2.imwrite(OUTPUT_PATH, result)

print("\n" + "=" * 60)
print("THERMAL HOTSPOT DETECTION RESULTS")
print("=" * 60)
print(f"Input image       : {IMAGE_PATH}")
print(f"Output image      : {OUTPUT_PATH}")
print(f"Threshold         : {HOTSPOT_THRESHOLD}%")
print(f"Minimum blob area : {MIN_BLOB_AREA} pixels")
print(f"\nDetected hotspots : {len(detected_hotspots)}")
print("-" * 60)

for hotspot in detected_hotspots:
    print(f"\nHotspot {hotspot['id']}")
    print(f"  Bounding box    : ({hotspot['x']}, {hotspot['y']}) {hotspot['width']} x {hotspot['height']} px")
    print(f"  Area            : {hotspot['area']} pixels")
    print(f"  Centroid        : ({hotspot['centroid'][0]:.1f}, {hotspot['centroid'][1]:.1f})")
    print(f"  Mean intensity  : {hotspot['mean_intensity']:.1f}/255")
    print(f"  Maximum intensity: {hotspot['max_intensity']:.1f}/255")
    if hotspot["temperature"] is not None:
        print(f"  Mean temperature: {hotspot['temperature']:.2f} °C")
        print(f"  Maximum temperature: {hotspot['max_temperature']:.2f} °C")
    print(f"  Configuration / confidence rating: {hotspot['confidence']:.1f}%")

print("\n" + "=" * 60)
print("Detection image saved successfully.")
print("=" * 60)
