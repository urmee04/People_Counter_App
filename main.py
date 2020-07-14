"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 
 6Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
   
    return parser

def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    #client = None
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client

def draw_boxes(frame, result, args, width, height):
    """
    Parse SSD output.
    :param frame: frame from camera/video
    :param result: list contains the data to parse ssd
    :return: person count and frame
    """
   
    current_count = 0
    for box in result[0][0]:
        conf = box[2] 
        
        # Draw bounding box for object when it's probability is more than
        #  the specified threshold
        if conf >= args.prob_threshold:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (239, 174, 0), 2)
            current_count = current_count + 1
           
    return frame, current_count
        
def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    
    #set variables for counting people
    
    last_count = 0
    total_count = 0
    start_time = 0
   
    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(args.model, args.device, args.cpu_extension)
    net_input_shape = infer_network.get_input_shape()

    ### TODO: Handle the input stream ###
    ### TODO: Read from the video capture ###
    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)
   
    # Grab the shape of the input 
    width = int(cap.get(3))
    height = int(cap.get(4))
    
    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        # Read the next frame
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        ### TODO: Pre-process the image ###
       
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
       
        ### TODO: Start asynchronous inference for specified request ###
        
        inf_start = time.time()
        infer_network.async_inference(p_frame)
        
        ### TODO: Wait for the result ###
        
        if infer_network.wait() == 0:
            result = infer_network.extract_output()
            det_time = time.time() - inf_start
       
        ### TODO: Update the frame to include detected bounding boxes
            frame, current_count = draw_boxes(frame, result, args, width, height)  
          
        ### TODO: Get the results of the inference request ###
            
            inf_time_message = "Inference time: {:.3f}ms"\
                               .format(det_time * 1000)
     
        
            cv2.putText(frame,
                    inf_time_message,
                    (15, 15),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.5, 
                    (239, 174, 0),
                    1)
            
        ### TODO: Extract any desired stats from the results ###
        ### TODO: Calculate and send relevant information on ###
        ### current_count, total_count and duration to the MQTT server ###
        ### Topic "person": keys of "count" and "total" ###
        ### Topic "person/duration": key of "duration" ###
         
            if current_count > last_count:
                start_time = time.time()
                total_count = total_count + current_count - last_count
                client.publish("person", json.dumps({"total": total_count}))
               
            if current_count < last_count:
                duration = int(time.time() - start_time)
              
            # Publish messages to the MQTT server
               
                client.publish("person/duration",
                               json.dumps({"duration": duration}))

            client.publish("person", json.dumps({"count": current_count}))
            
            last_count = current_count

            if key_pressed == 27:
                break

        ### TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()

        ### TODO: Write an output image if `single_image_mode` ###
        cv2.imwrite('output_image.jpg', frame)
    
    # Release the capture and destroy any OpenCV windows       
    cap.release()
    cv2.destroyAllWindows()
    
    # Disconnect from MQTT
    client.disconnect()
    
def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)
   # model = COUNTER_MODEL


if __name__ == '__main__':
    main()
