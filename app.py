import time
import numpy as np
import cv2 
from flask import Flask, render_template, Response, redirect, url_for
from serial.tools import list_ports
import time
import pydobot

# Get cameras function from utils.py in RETA
def get_cameras():
    available_cams = []
    for camera_idx in range(10):
        cap = cv2.VideoCapture(camera_idx)
        if cap.isOpened():
            available_cams.append(camera_idx)
            cap.release()
        else:
            # suppress warnings from cv2
            print ('\033[A' + ' '*158 +  '\033[A')
    return available_cams

# dobot
available_ports = list_ports.comports()
print(f'available ports: {[x.device for x in available_ports]}')
available_cams = get_cameras()
print("available cameras: ", available_cams)
port = available_ports[0].device # DOBOT PORT NAME: /dev/ttyACM0
device = pydobot.Dobot(port=port, verbose=False)
device.speed(velocity=60, acceleration=60)
time.sleep(5)

# App
app = Flask(__name__)
camera_side=cv2.VideoCapture(2)

class pos_keeper():
    def __init__(self, j1, j2, j3, j4):
        self.j1=j1
        self.j2=j2
        self.j3=j3
        self.j4=j4
        
Pos = pos_keeper(0, 0, 0, 0)

@app.context_processor
def context_processor():
    return dict(j1=Pos.j1, 
                j2=Pos.j2, 
                j3=Pos.j3, 
                j4=Pos.j4)

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

# def switch_color():
#     if 'red' in request.form:
#         low_color = np.array([161, 155, 84]) # Red in HSV format
#         high_color = np.array([179, 255, 255])
#     elif 'blue' in request.form:
#         low_color = np.array([95, 155, 84]) # Blue
#         high_color = np.array([130, 255, 255])
#     return render_template('index.html')

color = 0

def generate_frames( mask:"bool"=False):
    '''
    Feed function int to select camera for cv2 to open
    
    !!! Consider only allowing movements for odd numbered frames
        - Simple but hacky way to get movements to slow down which should make 
          it easier for the dobot to register objects in view
    '''
    global color
    camera=cv2.VideoCapture(0)
    # Make resolution simpler to boost performance
    camera.set(3, 480) # switch width from 640 to 480
    camera.set(4, 320) # switch height from 480 to 320

    _, frame = camera.read()
    rows, cols, _ = frame.shape # rows, cols, channels
    x_center = int((cols) / 2) # x_center of screen
    x_medium = int((cols) / 2) 
    y_center = int((rows) / 2) # y_enter of screen
    old_y_center = int((rows) / 2)
    y_medium = int((rows) / 2) 
    # Reset Dobot Magician Lite position

    j1, j2, j3, j4 = 0, 0, 0, 0
    first_phase=True # First phase consists of centering screen to object 
    # Once first phase is over (screen centered on object) dobot will then grab object
    
    device.rotate_joint(j1, j2, j3, j4)
    # Loop for colored-object tracking
    count = 0 # Counter for how many times arm converged object to center
    stream = True
    found = False
    scan = 0 # Once no objects appear as targets, increment by 1 to perform a total of 4 scans
    scan_frames = 3 # Frame buffer used for scanning
    
    # Refer to this for post for how to input colors in HSV format: https://stackoverflow.com/questions/10948589/choosing-the-correct-upper-and-lower-hsv-boundaries-for-color-detection-withcv

    # low_color = np.array([40, 155, 84]) # Green
    # high_color = np.array([80, 255, 255])
    # low_color = np.array([20, 150, 140]) # Yellow
    # high_color = np.array([50, 255, 255]) # WARNING: YELLOW REQUIRES GOOD/NEUTRAL LIGHTING

    frame_num = 0
    
    while stream:
        # Read the camera frame
        # device.speed(velocity=60, acceleration=60)
        success, frame = camera.read() # returns (1) bool if we get frames or not (2) the frame
        ret, corners = cv2.findChessboardCorners(frame, (7,6), None)
        if not success:
            break
        else:
            frame = cv2.blur(frame, (3,3))
            frame = frame[5:, 63:287] # crops frames so that top of mat can easily be adjusted to fit snugly in frame
            # HSV ex at https://stackoverflow.com/questions/47483951/how-to-define-a-threshold-value-to-detect-only-green-colour-objects-in-an-image/47483966#47483966
            hsv_frame = cv2.cvtColor(frame[:200], cv2.COLOR_BGR2HSV) # returns (480, 640, 3) np arrays
            if color % 2 == 0:
                low_color = np.array([161, 155, 84]) # Red in HSV format
                high_color = np.array([179, 255, 255])
            elif color % 2 == 1:
                low_color = np.array([95, 155, 84]) # Blue
                high_color = np.array([130, 255, 255])
            color_mask = cv2.inRange(hsv_frame, low_color, high_color) # I think this might be frame's masked counterpart
            contours, _ = cv2.findContours(color_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # Countours is a list of rank=3 nparrays of varying dimensions
            contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True) # Sort contours by area, largest to smallest
            # Guiding Quesion 1
            # Establish priority for "large" contours which are closest to center
            try:
                maxArea = cv2.contourArea(contours[0]) # Largest area
                count = 0

            except:# If no area can be found from the "try"
                maxArea = 0 
                found = True
                count += 1
                if count >= 20: # after 20 count of no blocks, switch color
                    color += 1
                    if color % 2 == 0:
                        print("picking up red blocks")
                    if color % 2 == 1:
                        print("picking up blue blocks")
                    count = 0
            error = 0.2
            largeContourPairs = []
            for contour in contours: # Compile large contours
                area = cv2.contourArea(contour)
                if area <= maxArea + maxArea and area >= maxArea - maxArea: # If contour is within area boundaries
                    (x, y, w, h) = cv2.boundingRect(contour)
                    x_medium = int((x + x + w) / 2) # middle line must be int since pixels are ints
                    y_medium = int((y + y + h) / 2) 
                    dist = np.sqrt(np.power(x_medium - 112, 2) + np.power(y_medium - old_y_center, 2)) # 2d distance calc for object centroid to center of screen
                    largeContourPairs.append((contour, dist)) # append a contour, dist pair
                    found = False
            largeContourPairs = sorted(largeContourPairs, key=lambda largeContourPairs : largeContourPairs[1]) # Sort by dist
            # Now create crosshair to home in on object
            x_160mm_pxdist = 110 # vertical pixel distance from top of screen to home x position on grid
            for cnt, dist in largeContourPairs: # iterate over contour frames
                (x, y, w, h) = cv2.boundingRect(cnt)
                x_medium = int((x + x + w) / 2) # middle line must be int since pixels are ints
                y_medium = int((y + y + h) / 2)
                r_medium = np.sqrt(np.power(x_medium - 112, 2) + np.power(y_medium - (x_160mm_pxdist + (250 * x_160mm_pxdist / 160)), 2)) # 2d distance calc from object centroid to approximately the pole of j1 rotation
                theta_medium = -np.degrees(np.arctan((x_medium - 112) / ((x_160mm_pxdist + (250 * x_160mm_pxdist / 160)) - y_medium))) # polar angle of object centroid with pole at approximately the pole of j1 rotation and polar axis being vertical line, left half is positive, right half negative
                print(x_medium, y_medium) # position of block in image coordinates
                if frame_num < 10:
                    x1, y1, z1, r1, j11, j21, j31, j41 = device.pose()
                    device.rotate_joint(1.3 * theta_medium, 0, 0, 0) # rotates j1 to ready arm for grab
                    time.sleep(1/60)
                    frame_num += 1
                elif frame_num == 10: # grabs after 10 frames
                    # converts the image coordinates to Dobot coordinates
                    to_x = 410 - ((160/x_160mm_pxdist) * y_medium)
                    to_y = (180/112) * (112 - x_medium)
                    print(to_x, to_y) # position of block in Dobot coordinates
                    device.move_to(x=to_x, y=to_y, z=0, r=r1, wait=True)
                    grab()
                    j1, j2, j3, j4 = 0, 0, 0, 0
                    frame_num = 0
                # print(x_medium, y_medium)
                # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                break # Just interested in first sorted (used to be on biggest, now it's on closest) rectangle/lines
            # Code for crosshair
            cv2.line(frame, (x_medium,0), (x_medium, 480), (0, 255, 0), 1)
            cv2.line(frame, (0, y_medium), (640, y_medium), (0, 255, 0), 1)
            font = cv2.FONT_HERSHEY_SIMPLEX

            try: # Some error occurs when adding dist line that I'm too lazy to fix lol
                # Error seems only to occur AFTER First phase is completed and grabbing centering begins (dist line disappears; highly noticeable from video)
                cv2.line(frame, (x_medium, y_medium), (112, int(x_160mm_pxdist + (250 * x_160mm_pxdist / 160))), (255, 0, 255), 1) # Dist line
                cv2.putText(img = frame, text="(r, theta): ",  org=(x_medium + 20,  y_medium), fontFace=font, fontScale=0.3, 
                            color=(0, 255, 255))
                cv2.putText(img = frame, text=f"({round(r_medium)}, {theta_medium:.3})",  org=(x_medium + 20,  y_medium + 10), fontFace=font, fontScale=0.3, 
                            color=(0, 255, 255))
                cv2.putText(img = frame, text="(x, y): ",  org=(x_medium + 20,  y_medium + 20), fontFace=font, fontScale=0.3, 
                            color=(0, 255, 255))
                cv2.putText(img = frame, text=f"({x_medium}, {y_medium})",  org=(x_medium + 20,  y_medium + 40), fontFace=font, fontScale=0.3, 
                            color=(0, 255, 255))
            except:
                cv2.putText(img = frame, text=f"DIST: X.XX px",  org=(112 + 20,  old_y_center), fontFace=font, fontScale=0.3, 
                            color=(0, 255, 255))
                pass
            ret, buffer = cv2.imencode(".jpg", frame) # Encodes frame into memory buffer
            frame = buffer.tobytes()
            output_frame = b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
            # m_ret, m_buffer = cv2.imencode(".jpg", color_mask) # Encodes masked frame into memory buffer
            # color_mask = m_buffer.tobytes()
            # output_color_mask = b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + color_mask + b'\r\n'
            yield(output_frame)

def grab():
    '''
    Frames are basically frozen during this,
    consider using multiprocess or something to have the 
    grabbing function run while frame generator passes 
    in order to yield frames while grabbing is occurring
    '''
    global color
    x, y, z, r, j1, j2, j3, j4 = device.pose()
    device.grip(False)
    device.move_to(x=x, y=y, z=-24, r=r)
    time.sleep(1)
    device.grip(True)
    time.sleep(1)
    device.move_to(x=x, y=y, z=15, r=r)
    if color % 2 == 0: # if block is red, rotate j1 to 80 deg
        device.rotate_joint(j1=80, j2=30, j3=0, j4=0)
    if color % 2 == 1: # if block is blue, rotate j1 to 70 deg
        device.rotate_joint(j1=70, j2=30, j3=0, j4=0)
    time.sleep(1)
    device.grip(False)
    time.sleep(1)
    device.suck(False)

def generate_frames_side():
    '''
    Feed function int to select camera for cv2 to open
    '''
    
    # Loop for colored-object tracking
    while True:
        # Read the camera frame
        success, frame = camera_side.read() # returns (1) bool if we get frames or not (2) the frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode(".jpg", frame) # Encodes frame into memory buffer
            frame = buffer.tobytes()
            output_frame = b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
            yield(output_frame)

@app.route('/video_feed')
def video_feed():
    
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(generate_frames(False),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_side')
def video_feed_side():
    return Response(generate_frames_side(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/video_feed2')
# def video_feed2():
#     return Response(generate_frames(2, True),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    port = 5000
    app.run(host='0.0.0.0', port=port, debug=False)
    device.rotate_joint(0,0,0,0)