import yaml
import numpy as np
import cv2

#-----------------------------------------------------------------------------------------------------------------
#       Declaring Files
#-----------------------------------------------------------------------------------------------------------------

# Video file decleration
fn = r"./datasets/video.mp4"    

# parking spots cordinates decleration             
fn_yaml = r"./datasets/video.yml"           

config = {'text_overlay': True,
          'parking_overlay': True,
          'parking_id_overlay': True,
          'parking_detection': True,
          'min_area_motion_contour': 60,
          'park_sec_to_wait': 80}

# openCV on video file          
cap = cv2.VideoCapture(fn)

video_info = {'fps':    cap.get(cv2.CAP_PROP_FPS),
              'width':  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
              'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
              'fourcc': cap.get(cv2.CAP_PROP_FOURCC),
              'num_of_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}

cap.set(cv2.CAP_PROP_POS_FRAMES, 1000)



#-----------------------------------------------------------------------------------------------------------------
#       Rescalling Window as per to required need
#-----------------------------------------------------------------------------------------------------------------
def rescale_frame(frame, percent):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)


#-----------------------------------------------------------------------------------------------------------------
#       Opening the spots points/coordinates file
#-----------------------------------------------------------------------------------------------------------------
with open(fn_yaml, 'r') as stream:
    parking_data = yaml.load(stream, Loader=yaml.FullLoader)

parking_contours = []
parking_bounding_rects = []
parking_mask = []

#-----------------------------------------------------------------------------------------------------------------
#       Converting those points into axis on the canvas
#-----------------------------------------------------------------------------------------------------------------
for park in parking_data:
    points = np.array(park['points'])
    rect = cv2.boundingRect(points)
    points_shifted = points.copy()
    points_shifted[:,0] = points[:,0] - rect[0]
    points_shifted[:,1] = points[:,1] - rect[1]
    parking_contours.append(points)
    parking_bounding_rects.append(rect)
    mask = cv2.drawContours(np.zeros((rect[3], rect[2]), dtype=np.uint8), [points_shifted], contourIdx=-1, color=255, thickness=-1, lineType=cv2.LINE_8)
    mask = mask==255
    parking_mask.append(mask)

parking_status = [False]*len(parking_data)
parking_buffer = [None]*len(parking_data)




#-----------------------------------------------------------------------------------------------------------------
#       Running the video 
#-----------------------------------------------------------------------------------------------------------------
video_cur_frame = 0
video_cur_pos = 0
errorcolor = []
while(cap.isOpened()):          # untill all frames are read
    spot = 0                    # available spots
    occupied = 0                # occupied spot

    video_cur_pos +=1 
    video_cur_frame +=1

    ret, frame = cap.read()
    if ret==False:
        print("Capture Error")
        break

    frame_blur = cv2.GaussianBlur(frame.copy(), (5,5), 3)
    frame_gray = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2GRAY)
    frame_out = frame.copy()

    #-----------------------------------------------------------------------------------------------------------------
    #       Parking detection algorithm
    #-----------------------------------------------------------------------------------------------------------------  
    if config['parking_detection']:
        for ind, park in enumerate(parking_data):
            points = np.array(park['points'])
            rect = parking_bounding_rects[ind]
            roi_gray = frame_gray[rect[1]:(rect[1]+rect[3]), rect[0]:(rect[0]+rect[2])]
            points[:,0] = points[:,0] - rect[0] 
            points[:,1] = points[:,1] - rect[1]
            status = np.std(roi_gray) < 30 and np.mean(roi_gray) > 50
            if status != parking_status[ind] and parking_buffer[ind]==None:
                parking_buffer[ind] = video_cur_pos
            elif status != parking_status[ind] and parking_buffer[ind]!=None:
                if video_cur_pos - parking_buffer[ind] > config['park_sec_to_wait']:
                    parking_status[ind] = status
                    parking_buffer[ind] = None
            elif status == parking_status[ind] and parking_buffer[ind]!=None:
                parking_buffer[ind] = None

    #-----------------------------------------------------------------------------------------------------------------
    #       Display on the parking spot
    #----------------------------------------------------------------------------------------------------------------- 
    if config['parking_overlay']:
        for ind, park in enumerate(parking_data):
            points = np.array(park['points'])
            if parking_status[ind]:
                color = (0,255,0)               #green
                spot = spot+1
            else:
                color = (0,0,255)               #red
                occupied = occupied+1
            moments = cv2.moments(points)
            centroid = (int(moments['m10']/moments['m00'])-3, int(moments['m01']/moments['m00'])+3)
            if park['id'] in errorcolor:
                cv2.drawContours(frame_out, [points], contourIdx=-1, color=(128,0,128), thickness=3, lineType=cv2.LINE_8)
                cv2.putText(frame_out, str(park['id']), (centroid[0]-10, centroid[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (128,0,128), 2, cv2.LINE_AA)
            else:
                cv2.drawContours(frame_out, [points], contourIdx=-1, color=color, thickness=1, lineType=cv2.LINE_8)
                cv2.putText(frame_out, str(park['id']), (centroid[0]-10, centroid[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1, cv2.LINE_AA)

                                    
    #-----------------------------------------------------------------------------------------------------------------
    #      Display on canvas for frame rate and spots booked 
    #-----------------------------------------------------------------------------------------------------------------  
    if config['text_overlay']:
        str_on_frame = "Frames: %d/%d" % (video_cur_frame, video_info['num_of_frames'])
        cv2.putText(frame_out, str_on_frame, (5,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,128,255), 1, cv2.LINE_AA)
        str_on_frame = "Free: %d Occupied: %d" % (spot, occupied)
        cv2.putText(frame_out, str_on_frame, (5,30), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,128,255), 1, cv2.LINE_AA)

    #-----------------------------------------------------------------------------------------------------------------
    #       Resizing the canvas to fit on the screen
    #----------------------------------------------------------------------------------------------------------------- 
    frame = rescale_frame(frame_out, percent=200)
    cv2.imshow('Spot Detection System ', frame)

    #-----------------------------------------------------------------------------------------------------------------
    #       Quit from the program
    #----------------------------------------------------------------------------------------------------------------- 
    k = cv2.waitKey(1)
    if k == 32:
        cv2.waitKey()
    if k == ord('q'):           # quit program
        break
    elif k == ord('c'):         # capture image
        cv2.imwrite('frame%d.jpg' % video_cur_frame, frame_out)

cap.release()
cv2.destroyAllWindows()
