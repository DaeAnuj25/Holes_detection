"""Replaceable camera/rover distance-estimation interface.

The current implementation intentionally reports placeholders; it does not
pretend pixel geometry is a metric measurement before calibration is supplied.
"""
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class DistanceEstimate:
    camera_m: Optional[float]
    rover_m: Optional[float]
    calibrated: bool = False
    method: str = 'placeholder'

    def as_dict(self) -> Dict[str, Any]:
        return {
            'camera_m': self.camera_m,
            'rover_m': self.rover_m,
            'calibrated': self.calibrated,
            'method': self.method,
        }


class DistanceEstimator:
    """Interface for a later pixel-to-ground-plane calibration model."""

    def estimate(self, bbox_xyxy, image_shape) -> DistanceEstimate:
        """Estimate distances for one box; bbox is x1,y1,x2,y2 in pixels.

        Replace this method with a calibrated mapping (for example, a lookup
        table or fitted ground-plane/pixel-position curve). Camera and rover
        reference frames should be handled independently in that mapping.
        """
        return DistanceEstimate(camera_m=None, rover_m=None)
