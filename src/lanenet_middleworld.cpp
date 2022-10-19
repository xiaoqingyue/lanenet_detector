#include "ros/ros.h"
#include "lanenet_detector/Lane.h" 
#include <sstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <Eigen/Dense>
   

void messageCallback(const lanenet_detector::Lane::ConstPtr& middle_line)
{
    int num= middle_line->worldmiddle_point.size();
     //std::cout<<num<<std::endl;

    Eigen::MatrixXf world_point =  Eigen::MatrixXf::Zero(num/3,3);    
    int x=0;
    for(int i=0;i<num/3;i++)
           for(int j=0;j<3;j++){
               world_point(i,j) = middle_line->worldmiddle_point[x];
               x+=1;
               }  

    std::cout<<"middle_world:"<<world_point<<std::endl;
    //ROS_INFO("y: %x",world_point);
}

int main(int argc, char **argv)
{
   
    ros::init(argc, argv, "listener");//初始化ros 节点名listener

    ros::NodeHandle n;

    ros::Subscriber sub = n.subscribe("Lane_middle", 10, messageCallback);

    ros::spin(); //一直授权订阅
    return 0;
}
