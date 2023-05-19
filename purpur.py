#!/usr/bin/env python3

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge,CvBridgeError


def purple(img):
    bridge = CvBridge()
    open=bridge.imgmsg_to_cv2(img)
    col=cv2.cvtColor(open,cv2.COLOR_BGR2RGB)
   
    
    #PURPUR

    new=cv2.cvtColor(open,cv2.COLOR_BGR2LAB)
    low=np.array([39,139,52])
    hig=np.array([255,255,119])
    mas=cv2.inRange(new,low,hig)
    contor,_=cv2.findContours(mas,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contor=sorted(contor, key=lambda x:cv2.contourArea(x),reverse=True)

    if len(contor) < 35:
        pass
    else:
        area=cv2.contourArea(contor[0])
        (x,y,w,h)=cv2.boundingRect(contor[0])
        cv2.rectangle(col,(x,y),(x+w,y+h),(0,255,0),2)
        # center1 = (float(x + (w/2)), float(y + (h/2)))
    
        cv2.circle(col,(int(x + (w/2)), int(y + (h/2))),1,(255,0,0),5)
        print(f"x = {int(x + (w/2))}, y = {int(y + (h/2))}")
        print(open.shape)
        print(len(contor))
           
    cv2.imshow('new',col)
    # cv2.imshow("01",mas)
    cv2.waitKey(1)
        
    

def main():
    rospy.init_node("opencv",anonymous=True)
    sub = rospy.Subscriber("/usb_cam/image_raw",Image,purple)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
    

if __name__=="__main__":
    main()
