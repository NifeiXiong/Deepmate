# DeepMate: a deep learning model for mouse mating behavior prediction

## 1. Background Introduction
Traditional manual observation methods in mouse behavioral research are time-consuming and labor-intensive, and are prone to subjective biases, leading to insufficient accuracy and consistency of data. Solely relying on video analysis methods struggles to capture subtle and rapid behavioral changes in mice, especially in complex behavioral scenarios, where video analysis suffers from inefficiency and low sensitivity. Our project primarily serves researchers in mouse behavioral studies and drug research by providing a more efficient and accurate behavioral recognition tool to support basic research and drug screening applications.

## 2. Project Overview

Our application detects mouse mating behavior through ultrasonic vocalizations (USVs) and video analysis. We introduced USVs because they are strongly correlated with animal emotions and social interactions, allowing for early and accurate detection of subtle behaviors. This innovative approach overcomes the limitations of traditional video analysis methods by utilizing the time features of the ultrasonic signals, improving the accuracy and reliability of behavior recognition. By combining multimodal analysis and advanced data processing technologies, this tool enhances the efficiency and objectivity of behavior recognition, providing support for mouse behavioral research, neuroscience, and drug screening. Expanding the model’s behavioral detection range is also a future goal.

## 3. Project Features

1. **Ultrasonic Signal Analysis**: Using innovative ultrasonic data processing techniques, we analyze mouse ultrasonic signals to identify and analyze ultrasonic features related to mating behavior.
2. **Video Data Analysis**: A highly efficient video analysis module has been designed to only detect necessary frames and track targets, reducing computational overhead and improving analysis speed.
3. **Model Scalability**: The tool allows for the training of custom ultrasonic models.
4. **Convenient Video Clipping**: Quickly clip specific segments, simplifying the video data processing workflow.
5. **Other model options**: In addition to the model built using the article, users can also choose different models to fit different data.

## 4. Installation Guide

### (1) System Requirements

- Operating System: Windows / macOS / Linux
- Python Version: Python 3.x (recommended 3.8 or above)

### (2) Install Dependencies

The project's dependencies are typically listed in the `requirements.txt` file. You can install all dependencies with the following command:
```bash
pip install -r requirements.txt



## 5.Usage Instructions

### (1)Run the Project

#### Startup Command
```bash
python mat_interface.py 
```
A simple graphical user interface (GUI) has been created,allowing you to click and select files for operation directly.

### (2)File and Parameter Explanation

├─USV_data (stores data for the ultrasonic model)
│ ├─input (input data: table data)
│ ├─model (ultrasonic model, can choose from pre-trained or custom-trained models)
│ └─output (output: prediction results)
├─video_data (stores data for the video model)
│ ├─input (input data: video files)
│ └─output (output data: prediction results and cropped videos)
├─weight (pre-trained weights required for the video model)

- **`--Select XXX file/folder`**: Select input/output files.
- **`--Set threshold`**: Set the ultrasonic data segmentation threshold for training the model.
- **`--run`**: Run the project.

### (3)Basic Usage Page Overview


Top left: Use the ultrasonic model for prediction
Bottom left: Train your own ultrasonic model
Top right: Crop video
Bottom right: Use video for prediction

### (4)Notes

- Video model input formats::`asf`, `avi`, `gif`, `m4v`, `mkv`, `mov`, `mp4`, `mpeg`, `mpg`, `ts`, `wmv`, `webm` ；

- Crop video data to the appropriate style before using the model for prediction;

- Ultrasonic data should be in table format: table files output by deepsqueak, with the feature`interval`added to the last row, calculated as the`Begin Time (s)`of the current row minus the`End Time (s)`of the previous row; if you want to train your own model, the corresponding feature`y`must be labeled;；

- Please ensure that the data file format is correct. The ultrasonic table file should include the columns:  `Score`, `Call Length (s)`, `Principal Frequency (kHz)`, `Low Freq (kHz)`, `High Freq (kHz)','Delta Freq (kHz)', 'Frequency Standard Deviation (kHz)`, `Slope (kHz/s)`, `Sinuosity`,`Mean Power (dB/Hz)`, `Tonality`, `Peak Freq (kHz)`, `interval`, `y`。

- For large datasets, it is recommended to run in an environment with a GPU.


## 6.Technologies

- The video model is primarily based on YOLOv8, with modifications. The object detection and tracking module was first trained, and then changes were made in the \ultralytics\trackers\track.py file, skipping frames to detect mating behavior in mice, saving the detected mating periods.

- The ultrasound model performs data segmentation, then trains and predicts using an LSTM model with attention mechanisms.


