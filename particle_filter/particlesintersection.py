#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import PointStamped, Point, PoseArray, Point32, PolygonStamped
from sensor_msgs.msg import PointCloud
from laser_scan_get_map import MapClientLaserScanSubscriber 
from particle_filter import ParticleFilter
from sklearn.neighbors import NearestNeighbors as KNN
import tf_conversions

particles = np.empty((300,3))
recive_particles = 0

class RobotFusion(object):

    def __init__(self, PI_s, PI_t, alpha = 0.5, Sm = 1.5):
        
        self.point_pub = rospy.Publisher('/estimated_pose', PointStamped, queue_size = 6)
        self.colud_point_pub = rospy.Publisher('/transformed_particles', PointCloud, queue_size = 20)
        self.other_robot_observation_pub = rospy.Publisher('/other_robot_observation', PointCloud, queue_size = 20)
        self.scope_pub = rospy.Publisher('/scope', PolygonStamped, queue_size = 10)
        self.scan = MapClientLaserScanSubscriber()
        self.Sm = Sm
        self.PI_s = PI_s
        self.PI_t = PI_t
        self.alpha = alpha
        self.nbrs = KNN(n_neighbors=1, algorithm='ball_tree').fit(self.scan.obs())

    def robot_tracking (self, TH = 0.3):
        '''
        mu_t = np.mean(self.PI_t[:,0:2], axis = 0)
        cov_t = np.cov(self.PI_t[:,0:2])
        
        max_x = mu_t [0] + np.sqrt(cov_t[0,0]) * self.Sm
        min_x = mu_t [0] - np.sqrt(cov_t[0,0]) * self.Sm
        max_y = mu_t [1] + np.sqrt(cov_t[1,1]) * self.Sm
        min_y = mu_t [1] - np.sqrt(cov_t[1,1]) * self.Sm
        '''
        max_x = np.max(self.PI_t[:,0])
        min_x = np.min(self.PI_t[:,0])
        max_y = np.max(self.PI_t[:,1])
        min_y = np.min(self.PI_t[:,1])

        poly = PolygonStamped()
        poly.header.frame_id = "map"
        poly.polygon.points = []
        x = np.array([max_x, max_x, min_x, min_x])
        y = np.array([max_y, min_y, min_y, max_y])
        #print x, y
        for i in range (4):   
            points = Point32()        
            points.x = x[i]
            points.y = y[i]
            points.z = 0
            poly.polygon.points.append(points)      
        self.scope_pub.publish(poly)

        mu_s = np.mean(self.PI_s, axis = 0) 
        z = self.scan.scan2cart(mu_s).T*self.scan.map.map.info.resolution
        z = z[z[:,0] < max_x]
        z = z[z[:,0] > min_x]
        z = z[z[:,1] < max_y]
        z = z[z[:,1] > min_y]
        

        if len(z)>4:
            
            ob = self.scan.obs()
            _, indices = self.nbrs.kneighbors(z)
            z_n = ob[indices].reshape(z.shape)
        
            z = z[np.linalg.norm(z-z_n,axis=1) > TH]
            if len(z)>4:
                self.mu_t_hat = np.mean(z, axis = 0)
                cov_t_hat = np.cov(z)  
                self.send_robot_estimated(self.mu_t_hat)
                self.send_other_robot_observation(z)
                #print z.shape

    def send_robot_estimated(self,mu_hat):
        #self.mu_t_hat point stemp topic /clicked_point

        point = PointStamped()
        point.header.frame_id = "map"
        point.point.x = mu_hat[0]
        point.point.y = mu_hat[1]
        point.point.z = 0
        self.point_pub.publish(point)

    def transfer(self):
        mu_s = np.mean(self.PI_s, axis = 0)
        V = self.mu_t_hat - mu_s[0:2]
        #print self.PI_s[:,0:2] + V
        return self.PI_s[:,0:2] + V

    def send_transfered_particles(self):
        particles = self.transfer()
        particle_pose = PointCloud()
        particle_pose.header.frame_id = 'map'
        particle_pose.header.stamp = rospy.Time.now()
        particle_pose.points = []
        
        for ii in range(len(particles)):
            point = Point()
            point.x = particles[ii,0]
            point.y = particles[ii,1]
            point.z = 0
            
            particle_pose.points.append(point)
        
        self.colud_point_pub.publish(particle_pose)

    def send_other_robot_observation(self,z):
        particle_pose = PointCloud()
        particle_pose.header.frame_id = 'map'
        particle_pose.header.stamp = rospy.Time.now()
        particle_pose.points = []
        
        for ii in range(len(z)):
            point = Point()
            point.x = z[ii,0]
            point.y = z[ii,1]
            point.z = 0
            
            particle_pose.points.append(point)
        
        self.other_robot_observation_pub.publish(particle_pose)


    def chernoff_fusion(self, x, PF_1, PF_2, alpha, beta):
        return ((np.sum(-beta*np.linalg.norm(PF_1[:,0:2]-x)**2))**alpha) * ((np.sum(-beta*np.linalg.norm(PF_2[:,0:2]-x)**2))**(1-alpha)) 

    def update_weight(self):
        W = np.empty(len(self.PI_s))
        for ii in range (len(self.PI_s)):
            W[ii] = self.chernoff_fusion(self.PI_s[ii,0:2], self.PI_s, self.PI_t, self.alpha, 8)
        W = W/np.sum(W)
        if np.sum(np.isnan(W)) > 0:
            W = np.ones_like(W)/len(self.PI_s)
        print "fusing"
        return W


def particles2fuse(msg,pf,fusion): # callback function from topic /particlecloud
    global particles
    global recive_particles
    recive_particles = 1
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

def main():
    global particles
    global recive_particles
    rospy.init_node('particle_filter', anonymous = True)
    PF_l = ParticleFilter(Np=300)
    PI_t = np.random.randn(300,3)
    fusion = RobotFusion(PF_l.particles, PI_t)
    particles2fuse_cb = lambda x: particles2fuse(x,PF_l,fusion)
    rospy.Subscriber('/particlecloud2fuse_in', PoseArray, particles2fuse_cb)
    r = rospy.Rate(5)
    #plt.ion()
    #fig = plt.figure()
    time_last= rospy.Time.now()
    
    while not rospy.is_shutdown():
        time_now = rospy.Time.now()
        r.sleep()
        fusion.PI_s = PF_l.particles
        fusion.robot_tracking()
        
        PF_l.prediction()
        PF_l.likelihood_fild()
        if np.abs(PF_l.update_TH()) > 0.1:
            
            #PF_l.simpel_likelihood()
            PF_l.resampling()
            PF_l.i_TH = 0.0
            print 'Updating particles...'
        PF_l.pub()
        if time_now.to_sec() - time_last.to_sec() > 2:
            PF_l.pub2fuse()
            time_last = time_now

        if recive_particles:
            fusion.PI_t = particles
            PF_l.weights = fusion.update_weight()
            PF_l.resampling()
            recive_particles = 0
            
        #mean =  np.zeros(3)
        #mean = np.mean(PF_l.particles,axis=0)
        fusion.send_transfered_particles()
        #M = PF_l.scan.loction_based(mean)

        r.sleep()
        #plt.imshow(-M+PF_l.scan.occupancy_grid)
       # fig.canvas.draw()

    rospy.spin()


if __name__ == "__main__":
    main()
