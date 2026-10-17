"""Detekcja tablic (YOLO), wyrównanie perspektywy i metryki.

detector, aligner = PlateDetector("best.pt"), PlateAligner()
for det in detector.detect(img):
    plate = aligner.align(img, det.box)
"""

from alpr.detection.align import AlignedPlate, PlateAligner
from alpr.detection.detector import Detection, PlateDetector

__all__ = ["AlignedPlate", "Detection", "PlateAligner", "PlateDetector"]
