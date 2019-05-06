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

particles = np.empty((200,3))
recive_particles = 0
E_x = np.empty(3)

def particles2fuse(msg,pf,fusion): # callback function from topic /particlecloud
    global particles
    global recive_particles
    global E_x
    particles = np.empty((len(msg.poses),3))
    for i in range(len(msg.poses)):
        particles[i,0] = msg.poses[i].position.x
        particles[i,1] = msg.poses[i].position.y
        angle = (
            msg.poses[i].orientation.x,
            msg.poses[i].orientation.y,
            msg.poses[i].orientation.z,
            msg.poses[i].orientation.w) 
        particles[i,2] = tf_conversions.transformations.euler_from_quaternion(angle)[2]
    
    if np.trace(np.cov(particles[:,0:2].T)) < 0.15:
        recive_particles = 1


def main():
    global particles
    global recive_particles
    rospy.init_node('particle_filter', anonymous = True)
    PF_l = ParticleFilter(Np=200)
    PI_t = np.random.randn(200,3)
    fusion = RobotFusion(PF_l.particles, PI_t)
    particles2fuse_cb = lambda x: particles2fuse(x,PF_l,fusion)
    rospy.Subscriber('/particlecloud2fuse_in', PoseArray, particles2fuse_cb)
    r = rospy.Rate(5)
    time_last= rospy.Time.now()
    
    while not rospy.is_shutdown():
        time_now = rospy.Time.now()
        r.sleep()
        fusion.PI_s = PF_l.particles
        
        PF_l.pub()
        E_x = np.mean(fusion.PI_s,axis=0)
        if time_now.to_sec() - time_last.to_sec() > 1:
            PF_l.pub2fuse()
            time_last = time_now

        if recive_particles:
            print "recived particles!"
            fusion.PI_t = particles
            d = np.linalg.norm(np.mean(particles[:,0:2],axis=0) - fusion.mu_t_hat)
            print "d= ", d
            if d < 0.7:
                PF_l.weights = fusion.update_weight()
                PF_l.resampling()
            recive_particles = 0
        fusion.robot_tracking()    
        fusion.send_transfered_particles()

    rospy.spin()


if __name__ == "__main__":
    main()
