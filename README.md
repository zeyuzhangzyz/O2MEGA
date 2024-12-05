### About the code

`Video_detect.py`, `Video_process.py`, `Performance.py`, and `Video.py` contain the functions for video processing.

`Faster_RCNN_predict.py` is a modified version of the predict function, adapted to save results in the Yolov5 format.

`Motion_feature_map_extract.py` is the function used to extract the motion vector matrix from the video.

`Group_select.py` and `Manager.py` are components of the algorithm.

Running `Execution_setting.py` will generate the algorithm's results and output them to the `output_directory`.



### Dataset

| **Camera**                                                  | **Length (s)** | **Description**                           |
| ----------------------------------------------------------- | -------------- | ----------------------------------------- |
| [Mobile1](https://www.youtube.com/watch?v=1EiC9bvVGnk)      | 9651           | Daytime drive in Seattle streets.         |
| [Mobile2](https://www.youtube.com/watch?v=7o5PYCeEo2I)      | 5968           | Drive around Kuwait City.                 |
| [Mobile3](https://www.youtube.com/watch?v=6tyFAtgy4JA)      | 2157           | Daytime drive through downtown Vancouver. |
| [Mobile4](https://www.youtube.com/watch?v=Cw0d-nqSNE8)      | 5064           | Drive through Los Angeles downtown.       |
| [Mobile5](https://www.youtube.com/watch?v=kOMWAnxKq58)      | 2961           | Drive through Chicago downtown.           |
| [Mobile6](https://ieeexplore.ieee.org/document/5995586)      | 297            | Vehicle cameras in different scenarios.   |
| [Fixed1](https://www.youtube.com/watch?v=nt3D26lrkho)       | 840            | Relaxed highway traffic near French Alps. |
| [Fixed2](https://www.youtube.com/watch?v=MNn9qKG2UFI&t=81s) | 306            | Urban traffic for detection and tracking. |
| [Fixed3](https://www.youtube.com/watch?v=6tyFAtgy4JA)       | 2048           | Highway traffic for object recognition.   |
| [Fixed4](https://doi.org/10.1109/PETS-WINTER.2009.5399556)  | 904            | Same intersection from different angles.  |

**Note:** *Fixed 4 and Mobile 6 have multiple camera sources, 8 and 9 respectively.*

The folder `motivation_source_video` contains the videos of Fixed4 and Mobile6 used in the study, while the folder `source_video_demo` contains 5-second slices of the Mobile1-5 and Fixed1-3 videos.

The `Combined_motion_features` matrix is too large, so only a 5-second segment is saved, and it has been processed through a 2D CNN for feature extraction.



