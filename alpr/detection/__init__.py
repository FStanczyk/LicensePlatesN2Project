"""Detekcja tablic (YOLO), przycinanie i wyrównanie perspektywy.

    from alpr.detection import load_model, detect, align_plate
    model = load_model("runs/detect/plates/weights/best.pt")
    for det in detect(model, img):
        aligned, poly, method = align_plate(img, det.box)

Moduły: yolo (model, detekcja, konfiguracja treningu), align (wyrównanie), metrics (ewaluacja).
"""

from alpr.detection.align import align_plate
from alpr.detection.yolo import Detection, detect, load_model

__all__ = ["Detection", "align_plate", "detect", "load_model"]
