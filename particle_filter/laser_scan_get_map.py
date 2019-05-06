#!/usr/bin/env python 

import rospy
from sensor_msgs.msg import LaserScan
from nav_msgs.srv import GetMap
import numpy as np
from matplotlib import pyplot as plt

class MapClientLaserScanSubscriber(object):

    def __init__(self):
        rospy.Subscriber('/scan',LaserScan,self.get_scan)
        static_map = rospy.ServiceProxy('static_map',GetMap)
        self.z = rospy.wait_for_message("/scan",LaserScan)
        #print self.z

        self.map = static_map()
        map_info = self.map.map.info
        map_width = np.array(map_info.width) # map width
        map_heghit = np.array(map_info.height) # map heghit
        self.occupancy_grid = np.transpose(np.array(self.map.map.data).reshape(map_width, map_heghit)) # map

    def get_scan(self, msg):  # callback function for LaserScan topic
        self.z = msg

    def scan2cart(self, robot_origin = [0,0,0]):
        map_info = self.map.map.info
        map_width = np.array(map_info.width) # map width
        map_heghit = np.array(map_info.height) # map heghit
        
        # the robot orientation in relation to the world [m,m,rad]
        mu_x = robot_origin[0]
        mu_y = robot_origin[1]
        theta = map_info.origin.orientation.z + robot_origin[2]  
        ####################################################### need to add the robot localisation!!!! 

        # converting the laserscan measurements from polar coordinate to cartesian and create matrix from them.
        n = len(self.z.ranges); i = np.arange(len(self.z.ranges))
        angle = np.zeros(n); x = np.zeros(n); y = np.zeros(n)
    
        angle = np.add(self.z.angle_min, np.multiply(i, self.z.angle_increment)) + theta
        x = np.multiply(self.z.ranges, np.cos(angle)) + mu_x
        y = np.multiply(self.z.ranges, np.sin(angle)) + mu_y

        x[~np.isfinite(x)] = -1
        y[~np.isfinite(y)] = -1

        x = x / map_info.resolution
        #x_r = np.abs(x_r)
        #x = x.astype (int)
        x[x > map_width] = -1
        x[x < -map_width] = -1
        y = y / map_info.resolution
        #y_r = np.abs(y_r)
        #y = y.astype (int)
        y[y < -map_heghit] = -1
        y[y > map_heghit] = -1

        Y = np.stack((x,y))

        return Y

    def obs(self):
        return np.argwhere(self.occupancy_grid == 100)*self.map.map.info.resolution + self.map.map.info.resolution*0.5 + self.map.map.info.origin.position.x
        
    def loction_based(self, y = [0,0,0]):
        Y = np.array([y[0] + self.scan2cart(y)[0,:],y[1] + self.scan2cart(y)[1,:]])
        map_info = self.map.map.info
        map_width = np.array(map_info.width) # map width
        map_heghit = np.array(map_info.height) # map heghit
        # the round values of x,y coordinate from the laserscan are the indexes of the matrix which contains 0 by default and 100 for detecting an object
        matrix = np.zeros (shape = (map_width,map_heghit))
        for ii in range (len(self.z.ranges)):
            a_1 = int(Y[0,ii])
            a_2 = int(Y[1,ii])
            if a_1 != -1 and a_2 != -1:
                matrix[int(Y[0,ii]),int(Y[1,ii])] = 100
        return matrix

if __name__ == "__main__":
    rospy.init_node('laser_scan', anonymous = True)
    StaticMap = rospy.ServiceProxy('static_map',GetMap)

    scan = MapClientLaserScanSubscriber()
    r = rospy.Rate(1)
    OC = scan.occupancy_grid

    plt.ion()
    fig = plt.figure()
    while not rospy.is_shutdown():
        M = scan.loction_based()

        r.sleep()
        plt.imshow(-M+OC)
        fig.canvas.draw()
        print('Drawing!')

    rospy.spin()
    pass
