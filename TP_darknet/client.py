#!/usr/bin/env python
# coding: utf-8

import socket
import cv2
import numpy as np
import sys
import time
from scipy.misc import imread
import darknet as dn

net = dn.load_net("/home/valentin/Documents/Vision/TP_darknet/darknet/cfg/yolov3-tiny.cfg", "/home/valentin/Documents/Vision/TP_darknet/darknet/yolov3-tiny.weights", 0)
meta = dn.load_meta("/home/valentin/Documents/Vision/TP_darknet/darknet/cfg/coco.data")

if(len(sys.argv) != 3):
    print("Usage : {} hostname port".format(sys.argv[0]))
    print("e.g.   {} 192.168.0.39 1080".format(sys.argv[0]))
    sys.exit(-1)


cv2.namedWindow("Image")

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
host = sys.argv[1]
port = int(sys.argv[2])
server_address = (host, port)

while(True):
    
    sent = sock.sendto("get", server_address)

    data, server = sock.recvfrom(65507)
    print("Fragment size : {}".format(len(data)))
    if len(data) == 4:
        # This is a message error sent back by the server
        if(data == "FAIL"):
            continue
    array = np.frombuffer(data, dtype=np.dtype('uint8'))
    frame = cv2.imdecode(array, 1)
    im = array_to_image(frame)
    dn.rgbgr_image(im)
    r = dn.detectfj(net, meta, im)
    print "test direct opencv (cv2) read for image :"
    print r

    for x in r:
        r= random.randint(0, 255)
        g= random.randint(0,255)
        b= random.randint(0,255)
        
        cv2.rectangle(frame,(int(x[2][0]-x[2][2]/2),int(x[2][1]+x[2][3]/2)),(int(x[2][0]+x[2][2]/2),int(x[2][1]-x[2][3]/2)), (r,g,b), 2)
        cv2.putText(frame,x[0],(int(x[2][0]-x[2][2]/2),int(x[2][1]+x[2][3]/2)),cv2.FONT_HERSHEY_SIMPLEX,1.5,(r,g,b))
        
        test = frame[int(x[2][1]-x[2][3]/2): int(x[2][1]+x[2][3]/2), int(x[2][0]-x[2][2]/2):int(x[2][0]+x[2][2]/2)]
        
        path = '/home/valentin/Documents/Vision/TP_darknet/images_2'
        cv2.imwrite(os.path.join(path , str(time.time())+ '_' +x[0]+'.jpg'), test)
         
    cv2.imshow("Image", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("The client is quitting. If you wish to quite the server, simply call : \n")
print("echo -n \"quit\" > /dev/udp/{}/{}".format(host, port))
