#!/usr/bin/env python 

import rospy
import numpy as np
from geometry_msgs.msg import PointStamped, Point, PoseArray, Point32, PolygonStamped
from sensor_msgs.msg import PointCloud
from laser_scan_get_map import MapClientLaserScanSubscriber 
from particle_filter import ParticleFilter
from particlesintersection import RobotFusion
from sklearn.neighbors import NearestNeighbors as KNN
import tf_conversions

def main():

    rospy.init_node('ParticleFilter', anonymous = True)
    PF_l = ParticleFilter(Np=300)
    r = rospy.Rate(5)
    
    while not rospy.is_shutdown():
        r.sleep()

        PF_l.pub()

    rospy.spin()


if __name__ == "__main__":
    main()