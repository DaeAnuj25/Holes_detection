from pothole_detection.distance import DistanceEstimator


def test_uncalibrated_estimator_is_explicit_placeholder():
    estimate = DistanceEstimator().estimate([1, 2, 30, 40], (480, 640))
    assert estimate.camera_m is None
    assert estimate.rover_m is None
    assert estimate.calibrated is False
    assert estimate.method == 'placeholder'
