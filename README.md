# Pothole Detection for ROS 2 Jazzy

ROS 2 image-to-detection node using `new_model.pt`, with grayscale-first inference (maximum 5 FPS), JSON detections on a configurable topic, replaceable distance-estimation logic, and a local Linux live dashboard.

Start with the complete fresh-Ubuntu setup and operating guide in [`tutorial.txt`](tutorial.txt). For camera-free testing, `ros2 run pothole_detection test_image_publisher --ros-args -p source:=/path/to/clip.mp4` publishes looping video frames to `/camera/image_raw` by default. The dashboard defaults to `http://127.0.0.1:8080`.

**Distance output is currently uncalibrated (`null`).** It must not be used for rover navigation until the estimator is calibrated and validated.
