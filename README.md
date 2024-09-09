## Update
__(29/7/2020)__
- Rename `utils.py` to  `local_utils.py` to avoid conflicit with default Python library `utils.py`.
- Replace error `index out of range` to `No License plate is founded!`.
- In case error `No License Plate is founded!` popped up, try to adjust Dmin from `get_plate()` function. Keep in mind that larger Dmin means more higly the plate information is lost.

## [Read the series on Medium](https://medium.com/@quangnhatnguyenle/detect-and-recognize-vehicles-license-plate-with-machine-learning-and-python-part-1-detection-795fda47e922)
- Part 1: [Detection License Plate with Wpod-Net](https://medium.com/@quangnhatnguyenle/detect-and-recognize-vehicles-license-plate-with-machine-learning-and-python-part-1-detection-795fda47e922)
- Part 2: [Plate character segmentation with OpenCV](https://medium.com/@quangnhatnguyenle/detect-and-recognize-vehicles-license-plate-with-machine-learning-and-python-part-2-plate-de644de9849f)
- Part 3: [Recognize plate license characters with OpenCV and Deep Learning](https://medium.com/@quangnhatnguyenle/detect-and-recognize-vehicles-license-plate-with-machine-learning-and-python-part-3-recognize-be2eca1a9f12)

## Tools and Libraries
- keras
- tensorflow
- opencv-python==3.4.2.16
- pip>=19.2.3
- setuptools>=41.2.0
- django>=3.0.0
- psycopg2
- pillow
- matplotlib

### documentation link
[Lin Lin Hlaing](https://github.com/linlinhlaing/detect_car/blob/main/DocumentPlateProject.docx(lin%20lin%20hlaing).pdf)
## Credit
[sergiomsilva](https://github.com/sergiomsilva/alpr-unconstrained)
