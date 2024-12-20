import numpy as np
import cv2
import os 

# Load the video
cap = cv2.VideoCapture(os.path.join(os.path.dirname(__file__),'vtest.avi'))

# Parameters for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                          qualityLevel = 0.3,
                          minDistance = 7,
                          blockSize = 7 )

# Parameters for Lucas-Kanade optical flow
lk_params = dict( winSize  = (15,15),
                    maxLevel = 2,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

color = np.random.randint(0,255,(100,3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
if not ret:
    print("Error reading video")
    cap.release()
    cap.destroyAllWindows()
    exit()

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    tick_start = cv2.getTickCount()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    good_new = p1[st==1]
    good_old = p0[st==1]

    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        a, b, c, d = map(int, [a, b, c, d])
        mask = cv2.line(mask, (a,b), (c,d), color[i].tolist(), 2)
        frame = cv2.circle(frame, (a,b), 5, color[i].tolist(), -1)
    
    img = cv2.add(frame, mask)

    tick_end = cv2.getTickCount()
    time_per_frame = (tick_end - tick_start) / cv2.getTickFrequency()

    print("Time per frame: %.2f [ms]" % (time_per_frame * 1000))

    cv2.imshow('frame', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)

cap.release()
cv2.destroyAllWindows()


    
