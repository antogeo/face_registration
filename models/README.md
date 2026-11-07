# Models

`face_detection_yunet_2023mar.onnx` is the YuNet face detector used by `detect.py`.
It comes from the [OpenCV model zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) (MIT license).

This model has a **fixed input size**. `detect.py` resizes and pads every image to
`MODEL_INPUT_SIZE` before detection and scales the results back.

To download it again:

```bash
curl -L -o models/face_detection_yunet_2023mar.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

The file should be about 230 KB.
