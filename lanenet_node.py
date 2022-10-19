#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# @Author  : Luo Yao
# @Modified  : AdamShan
# @Original site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : lanenet_node.py


import time
import math
import tensorflow as tf
import numpy as np
import cv2

from lanenet_model import lanenet
from lanenet_model import lanenet_post_process,lanenet_postprocess_perspective
from config import global_config

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError
from lanenet_detector.msg import Lane
from lanenet_detector.msg import Curve
from lanenet_detector.msg import Lane_Image


CFG = global_config.cfg


class lanenet_detector():
    def __init__(self):
        self.image_topic = rospy.get_param('~image_topic')
        self.output_image = rospy.get_param('~output_image')
        self.output_lane = rospy.get_param('~output_lane')
        self.lane_curve_topic = rospy.get_param('~lane_curve_topic')
        self.lane_middle_topic = rospy.get_param('~lane_middle_topic')
        self.weight_path = rospy.get_param('~weight_path')
        # self.use_gpu = rospy.get_param('~use_gpu')
        self.lane_image_topic = rospy.get_param('~lane_image_topic')

        self.init_lanenet()  #初始化网络模型

        self.bridge = CvBridge()
        sub_image = rospy.Subscriber(self.image_topic, Image, self.img_callback, queue_size=1)
        self.pub_image = rospy.Publisher(self.output_image, Image, queue_size=1)
        self.pub_laneimage = rospy.Publisher(self.lane_image_topic, Lane_Image, queue_size=1)
        self.pub_lanecurve = rospy.Publisher(self.lane_curve_topic, Curve, queue_size=10)
        self.pub_lanemiddle = rospy.Publisher(self.lane_middle_topic, Lane, queue_size=10)

    
    def init_lanenet(self):
        '''
        initlize the tensorflow model
        '''

        self.input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')
        phase_tensor = tf.constant('test', tf.string)
        net = lanenet.LaneNet(phase=phase_tensor, net_flag='vgg')
        self.binary_seg_ret, self.instance_seg_ret = net.inference(input_tensor=self.input_tensor, name='lanenet_model')

        # self.cluster = lanenet_cluster.LaneNetCluster()
        self.postprocessor = lanenet_postprocess_perspective.LaneNetPostProcessor()

        saver = tf.train.Saver()
        # Set sess configuration
        # if self.use_gpu:
        #     sess_config = tf.ConfigProto(device_count={'GPU': 1})
        # else:
        #     sess_config = tf.ConfigProto(device_count={'CPU': 0})
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
        sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
        sess_config.gpu_options.allocator_type = 'BFC'

        self.sess = tf.Session(config=sess_config)
        saver.restore(sess=self.sess, save_path=self.weight_path)
    
    def img_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            #cv_image = cv2.flip(cv_image, -1)
        except CvBridgeError as e:
            print(e)    

        original_img = cv_image.copy()
        t_start = time.time()

        resized_image = self.preprocessing(cv_image)   #图像归一化 
        # 车道线检测+图像后处理
        postprocess_result= self.inference_net(resized_image, original_img)

        if postprocess_result['lane_lines'] is None:
            out_img_msg = self.bridge.cv2_to_imgmsg(postprocess_result['source_image'], "bgr8")
            self.pub_image.publish(out_img_msg)  

            t_cost = 1/(time.time() - t_start)    
            rospy.loginfo('Single imgae inference cost time: {:.5f}s'.format(t_cost))
            cv2.namedWindow("lane_image")
            cv2.imshow("lane_image", postprocess_result['source_image'])
            cv2.waitKey(25)

            # 发布曲率+中线坐标
            rate = rospy.Rate(10) # 10hz                     
            # lane_number=0
            # lane_a=[0]
            # lane_b=[0]
            # lane_c=[0]
            # lane_curve=[0] 

            lane_middcurv=0            
            lane_middle =[0] 
            world_point = [0] 
        
            # self.pub_lanecurve.publish(Curve(lane_number,lane_a,lane_b,lane_c,lane_curve))
            self.pub_lanemiddle.publish(Lane(lane_middle,lane_middcurv,world_point))

            rate.sleep()            
        else:
            # 发布图像
            out_img_msg = self.bridge.cv2_to_imgmsg(postprocess_result['result_image'], "bgr8")
            self.pub_image.publish(out_img_msg)                   
           
            t_cost = 1/(time.time() - t_start)  
            rospy.loginfo('Single imgae inference cost time: {:.5f}s'.format(t_cost))
            
            cv2.namedWindow("lane_image")
            cv2.imshow("lane_image", postprocess_result['result_image'])
            cv2.waitKey(25)

            # 发布曲率+中线坐标
            rate = rospy.Rate(10) # 10hz            
            
            # #车道线系数+曲率+车道线个数
            # lane_lines=postprocess_result['lane_lines']

            # lane_number=0
            # lane_a=[]
            # lane_b=[]
            # lane_c=[]
            # lane_curve=[]       

            # for line in lane_lines:        
            #     lane_a.append(line.params[0])
            #     lane_b.append(line.params[1])
            #     lane_c.append(line.params[2])   
            #     lane_curve.append(line.curvered)
            #     lane_number += 1                  
           
            # self.pub_lanecurve.publish(Curve(lane_number,lane_a,lane_b,lane_c,lane_curve))

            #中位线系数
            middle_line=postprocess_result['middle_line'].params
           
            lane_middle=[]

            for i in range(0,3):
                lane_middle.append(middle_line[i])            

            #中位线曲率
            lane_middcurv = postprocess_result['middle_curvatures']  

            # 中线世界坐标点
            worldmiddle_point=postprocess_result['middleline_world']
            world_point = []
            for i in range(0,len(worldmiddle_point),20):
                for j in worldmiddle_point[i]:
                    world_point.append(j)

            
            self.pub_lanemiddle.publish(Lane(lane_middle,lane_middcurv,world_point))

            rate.sleep()
            

    def preprocessing(self, img):
        image = cv2.resize(img, (512, 256), interpolation=cv2.INTER_LINEAR)
        image = image / 127.5 - 1.0       
        return image

    def inference_net(self, img, original_img):
        binary_seg_image, instance_seg_image = self.sess.run([self.binary_seg_ret, self.instance_seg_ret],
                                                        feed_dict={self.input_tensor: [img]})

        postprocess_result = self.postprocessor.postprocess(
            binary_seg_result=binary_seg_image[0],
            instance_seg_result=instance_seg_image[0],
            source_image=original_img
        )

        if postprocess_result['lane_lines'] is None:
            return postprocess_result
        else:
         # 计算曲率半径和车道中线
            lanelines_result = self.postprocessor.postprocess_curv_middle(postprocess_result['lane_lines'],
                                                           postprocess_result['source_image'])
        
            return lanelines_result     


    def minmax_scale(self, input_arr):
        """
        :param input_arr:
        :return:
        """
        min_val = np.min(input_arr)
        max_val = np.max(input_arr)

        output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

        return output_arr  


if __name__ == '__main__':
    # init args
    rospy.init_node('lanenet_node')
    lanenet_detector()
    rospy.spin()