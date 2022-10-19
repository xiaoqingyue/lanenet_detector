#!/usr/bin/env python2
# -*- coding: UTF-8 -*-

from lanenet_detector.msg import Curve
from lanenet_detector.msg import Lane
import rospy
import numpy as np

def lanelane_callback(lane_line):
   #传过来的所有msg为data，其中data.后边的变量看msg文件写，比如我的是float32[] data,那么我就是data.data
    rospy.loginfo(lane_line.lane_num)
    rospy.loginfo(lane_line.Curve)
    rospy.loginfo(lane_line.a)

def middleline_callback(middle_line):
    #传过来的所有msg为data，其中data.后边的变量看msg文件写，比如我的是float32[] data,那么我就是data.data
    # rospy.loginfo(middle_line.middle_line)
    # rospy.loginfo(middle_line.middle_curvatures)

    world_point = np.zeros((len(middle_line.worldmiddle_point)/3, 3), dtype=np.float64)
    x = 0
    for i in range(0, len(middle_line.worldmiddle_point)/3):
        for j in range(0, 3):
            world_point[i][j] = middle_line.worldmiddle_point[x]
            x += 1

    rospy.loginfo(world_point)



def listener():
    rospy.init_node('listener', anonymous=True)
    #rospy.Subscriber('Lane_curve', Curve, lanelane_callback)
    rospy.Subscriber('Lane_middle', Lane, middleline_callback)
    rospy.spin()


if __name__ == '__main__':
    listener()
