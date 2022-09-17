# Parking Spot Detection System
Parking spot detection system using openCV image processing to detect vehicles in a parking spot.

### How does it works
*   A prerecorded video file 
*   A yaml file with all the coordinates of the parking spots
*   openCV library used to detect a vehicle in a parking spot
*   If vehicle found, the parking spot is booked i.e. red, otherwise available i.e. green
---
### Installation
Installing the required packages
```
pip install -r requirements.txt
```


To run this program locally:
```
python main.py
```

## Working

As soon as the program starts, a video starts running with the predefined spots. These spots are imported from the video.yml file with contains the coordinates on the screen. 

The model detects which spots are booked and with spots are available and color codes it with red for booked and green for available.

<img src="https://github.com/Mujtaba1399/Parking-Spot-Detection-System/blob/main/datasets/spotsDetected.jpg">

##
As soon as a vehicle is parked at an available spot, the models detects its and changes the color of that spot from green to red, identifying it as booked. 

<img src="https://github.com/Mujtaba1399/Parking-Spot-Detection-System/blob/main/datasets/vehicleDetected.jpg">


