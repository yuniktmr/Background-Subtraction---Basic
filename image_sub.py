import numpy as np 
import argparse 
import imutils  
import cv2  
# construct the argument parser and parse the arguments 
ap = argparse.ArgumentParser() 
ap.add_argument("-b", "--bg", required=True,help="path to background image") 
ap.add_argument("-f", "--fg", required=True, help="path to foreground image") 
args = vars(ap.parse_args())
bg = cv2.imread(args["bg"]) 
fg = cv2.imread(args["fg"])
bgGray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY) 
fgGray = cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY)

sub = bgGray.astype("int32") - fgGray.astype("int32")
sub = np.absolute(sub).astype("uint8")

ret, thresh = cv2.threshold(sub, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

cv2.imshow("thresh", thresh)

thresh = cv2.erode(thresh, None, iterations = 1)
thresh = cv2.dilate(thresh, None, iterations = 1)
#cv2.imshow("morphed", thresh)
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cnts = imutils.grab_contours(cnts)
(minX, minY) = (np.inf, np.inf)
(maxX, maxY) = (-np.inf, -np.inf)
total = 0
for c in cnts:
    (x,y,w,h) = cv2.boundingRect(c)
    if w>20 and h>20:
        minX = min(minX, x)
        minY = min(minY, y)
        maxX = max(maxX, x+w-1)
        maxY = max(maxY, y+h-1)
        #print("min")
        #print((minX,minY))
       # print("max")
       # print((maxX,maxY))
        total =total +1
#print("finmin")
#print((minX,minY))
#print("finmax")
#print((maxX,maxY))

cv2.rectangle(fg, (minX, minY), (maxX, maxY), (0,255,0), 2)
cv2.imshow("Output", fg)
cv2.waitKey(0)
