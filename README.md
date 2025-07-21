# Facial Recognition Attendance System

This is a desktop application built with Python and Tkinter that uses facial recognition to automate the process of taking attendance.

## Features

* Register new users by capturing their face images.
* Train a facial recognition model on the registered users.
* Take attendance in real-time using a webcam.
* View the daily attendance log within the application.

## Requirements

* Python 3
* OpenCV
* Pillow
* Numpy
* Pandas

## Setup and Installation

1.  **Clone the repository (or download the files):**
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```

2.  **Install the required libraries:**
    It's recommended to use a virtual environment.
    ```bash
    pip install opencv-python numpy pandas pillow
    ```

3.  **Download the Haar Cascade file:**
    You need `haarcascade_frontalface_default.xml` for face detection. Download it and place it in the same folder as `app.py`. You can get it from [here](https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml).

## How to Run

1.  **Run the application:**
    ```bash
    python app.py
    ```

2.  **Register a User:**
    * Enter a numeric ID and a Name in the registration section.
    * Click "Take Images" and follow the on-screen instructions.
    * Click "Save Profile" to train the model. You will be asked to set or enter a password.

3.  **Take Attendance:**
    * Click "Take Attendance".
    * The camera will open and recognize registered faces.
    * Attendance is logged automatically. Press 'q' to close the camera window.