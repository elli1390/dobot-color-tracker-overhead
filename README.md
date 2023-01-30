# dobot-color-tracker-overhead
Flask app that streams feed from an overhead camera at a specific position and uses OpenCV to drive colored object tracking with a Dobot Magician Lite
## Summary
This repo contains a Flask app (runnable with something like ```$ python3 app.py```) that is runnable on a Raspberry Pi server which is connected to a Dobot and a Dobot USB camera (included with Dobot kits) overhead. This app is very rough and only operational at a specific camera position that is documented [here](https://github.com/elli1390/dobot-color-tracker-overhead/blob/main/Overhead%20Color%20Picking%20Exhibit%20Guide.pdf). This app is a modified version of [a colored object sorting (computer vision) app](https://github.com/supertechft/dobot-color-tracker) by nick33333 and Prince25 that utilizes a claw POV camera. OpenCV is used to drive the computer vision aspects of this app. The pydobot library is used to drive a Dobot connected via USB.

### Documentation forthcoming
