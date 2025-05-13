# Self-_driving_car-_simulation_projects
This project is a simulation-based self-driving car system that integrates Computer Vision, Deep Learning, and Natural Language Processing (NLP) to mimic autonomous vehicle behavior. The system is capable of object detection, lane detection, steering angle prediction, obstacle-based control, and distance estimation — all within a simulated environment.

🔍 Features
YOLOv8 Object Detection: Detects vehicles, pedestrians, and road signs in real time.

Lane Detection: Identifies lane boundaries to help maintain safe navigation.

Steering Angle Prediction: Predicts steering wheel rotation using a CNN model trained on road images.

Obstacle Control Module: Automatically stops or slows the car if an object is detected ahead.

Distance Estimation: Calculates object distance using image geometry.

Voice Commands (NLP): Processes basic natural language commands using an LLM for vehicle control (start, stop, etc.).

🛠️ Tech Stack
Python, OpenCV, NumPy, Pandas

YOLOv8 (Ultralytics) for object detection

CNN (TensorFlow/Keras) for steering prediction

Whisper for speech-to-text conversion

LangChain + OpenAI GPT for NLP-based command processing

Google Colab and VS Code for development and training

📊 Dataset
Road image datasets and driving behavior data from Kaggle and open-source repositories.

🎯 Goal
To simulate a modular self-driving system that can be extended to real-world AV (Autonomous Vehicle) scenarios for research and educational purposes.

📁 Modules
object_detection/

lane_detection/

steering_prediction/

voice_control/

distance_estimation/

Sample Output :

![image alt](https://github.com/Kaaviyasuresh/Self-_driving_car-_simulation_projects/blob/deb534f6c9a4c1cb6a61276b327575921c7b0045/SELF%20driving%20car%20iamgeoutput.jpg)

![image alt](774432c9b77249abdc565ec4326c508be312be51)

