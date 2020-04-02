# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 04:07:58 2017

@author: Isaac Watts
"""

import pickle
import numpy as np
from sklearn.cluster import KMeans
import sklearn.decomposition as decomp
import matplotlib.pyplot as plot
from mpl_toolkits.mplot3d import Axes3D


def axisEqual(data,ax):
    #Sets the scale of all axises to be equal
    #code modified from http://bit.ly/1kdHf2i
    X = data[:,0]
    Y = data[:,1]
    Z = data[:,2]
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
    Zb = 0.5*max_range*np.mgrid[0:3:2,0:3:2,0:3:2][2].flatten() + 0.5*(Z.max()+Z.min())
    
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')
    
    
class Box:
    def __init__(self):
        self.corners = []
        self.zmin = []
        self.zmax = []
        self.collideCheck = False
        self.n = 0
        
    def findCorners(self,cluster,label,n):
    # Finds the point coordinates of the box corners for each cluster
        self.n = n
        for i in range(n):
            pca = decomp.PCA(n_components=2)
            fitCluster = np.vstack((cluster[label == i][:,0],cluster[label == i][:,1])).T
            pca.fit(fitCluster)
            clusterTrans = pca.transform(fitCluster)
            
            boxTrans = np.array([[clusterTrans[:,0].min(),clusterTrans[:,1].min()],
                                  [clusterTrans[:,0].min(),clusterTrans[:,1].max()],
                                  [clusterTrans[:,0].max(),clusterTrans[:,1].max()],
                                  [clusterTrans[:,0].max(),clusterTrans[:,1].min()]
                                ])
            
            box = pca.inverse_transform(boxTrans)
            
            self.corners.append(box)
            self.zmin.append(cluster[:,2].min())
            self.zmax.append(cluster[:,2].max())
        
    def drawBoxes(self,ax):
    # Plots the boxes using the calculated corners for each cluster
        for i in range(self.n):   
            ax.plot([self.corners[i][0,0], self.corners[i][1,0]], [self.corners[i][0,1],self.corners[i][1,1]],zs=[self.zmax[i],self.zmax[i]],color="b")
            ax.plot([self.corners[i][0,0], self.corners[i][3,0]], [self.corners[i][0,1],self.corners[i][3,1]],zs=[self.zmax[i],self.zmax[i]],color="b")
            ax.plot([self.corners[i][0,0], self.corners[i][0,0]], [self.corners[i][0,1],self.corners[i][0,1]],zs=[self.zmax[i],self.zmin[i]],color="b")
                
            ax.plot([self.corners[i][2,0], self.corners[i][1,0]], [self.corners[i][2,1],self.corners[i][1,1]],zs=[self.zmax[i],self.zmax[i]],color="b")
            ax.plot([self.corners[i][2,0], self.corners[i][3,0]], [self.corners[i][2,1],self.corners[i][3,1]],zs=[self.zmax[i],self.zmax[i]],color="b")
            ax.plot([self.corners[i][2,0], self.corners[i][2,0]], [self.corners[i][2,1],self.corners[i][2,1]],zs=[self.zmax[i],self.zmin[i]],color="b")
            
            ax.plot([self.corners[i][1,0], self.corners[i][0,0]], [self.corners[i][1,1],self.corners[i][0,1]],zs=[self.zmin[i],self.zmin[i]],color="b")
            ax.plot([self.corners[i][1,0], self.corners[i][2,0]], [self.corners[i][1,1],self.corners[i][2,1]],zs=[self.zmin[i],self.zmin[i]],color="b")
            ax.plot([self.corners[i][1,0], self.corners[i][1,0]], [self.corners[i][1,1],self.corners[i][1,1]],zs=[self.zmin[i],self.zmax[i]],color="b")
                
            ax.plot([self.corners[i][3,0], self.corners[i][0,0]], [self.corners[i][3,1],self.corners[i][0,1]],zs=[self.zmin[i],self.zmin[i]],color="b")
            ax.plot([self.corners[i][3,0], self.corners[i][2,0]], [self.corners[i][3,1],self.corners[i][2,1]],zs=[self.zmin[i],self.zmin[i]],color="b")
            ax.plot([self.corners[i][3,0], self.corners[i][3,0]], [self.corners[i][3,1],self.corners[i][3,1]],zs=[self.zmin[i],self.zmax[i]],color="b")
            
    def collisionDetect(self):
    # Performs collision detection using the method described here https://bit.ly/39H7f7Q
        for i in range(self.n):
            for j in range(self.n - (i+1)):
                axis1 = [self.corners[i][1,0] - self.corners[i][2,0] , self.corners[i][1,1] - self.corners[i][2,1]]
                axis2 = [self.corners[i][1,0] - self.corners[i][2,0] , self.corners[i][1,1] - self.corners[i][2,1]]
                axis3 = [self.corners[i+j+1][0,0] - self.corners[i+j+1][3,0] , self.corners[i+j+1][0,1] - self.corners[i+j+1][3,1]]
                axis4 = [self.corners[i+j+1][0,0] - self.corners[i+j+1][1,0] , self.corners[i+j+1][0,1] - self.corners[i+j+1][1,1]]
                
                dotProductA = []
                dotProductB = []
                
                for s in [axis1, axis2, axis3, axis4]:
                    dotProductA = []
                    dotProductB = []
                    
                    for k in range(4):
                        projAx = ((self.corners[i][k,0] * s[0] + self.corners[i][k,1] * s[1]) / (s[0]**2+ s[0]**2)) * s[0]
                        projAy = ((self.corners[i][k,0] * s[0] + self.corners[i][k,1] * s[1]) / (s[0]**2 + s[0]**2)) * s[1]
                        
                        projBx = ((self.corners[i+j+1][k,0] * s[0] + self.corners[i+j+1][k,1] * s[1]) / (s[0]**2 + s[0]**2)) * s[0]
                        projBy = ((self.corners[i+j+1][k,0] * s[0] + self.corners[i+j+1][k,1] * s[1]) / (s[0]**2 + s[0]**2)) * s[1]
                        
                        dotProductA.append(projAx * s[0] + projAy * s[1])
                        dotProductB.append(projBx * s[0] + projBy * s[1])
                    
                    if min(dotProductB) > max(dotProductA) or max(dotProductB) < min(dotProductA):
                        self.collideCheck = False
                        break
                    
                else:
                    self.collideCheck = True
                    break
                
            if self.collideCheck == True:
                break
        
        return self.collideCheck
    
   
data = pickle.load(open("Project01.pkl", "rb"))
n = 2
collide = False

fig = plot.figure()
ax = fig.gca(projection='3d')

# Detects n number of clusters in the data using KMeans in sklearn then determines if any boxes are colliding.
# If any boxes are found to be colliding, n is increased by 1 and the process is run until nothing collides.
while True:
    box = Box()
    clusters = KMeans(n_clusters=n)
    clusters.fit(data)
    box.findCorners(data,clusters.labels_,n)
    collide = box.collisionDetect()
    del(box)
    if collide == True:
        n = n + 1
    else:
        break

box = Box()
clusters = KMeans(n_clusters=n)
clusters.fit(data)
box.findCorners(data,clusters.labels_,n)

for i in range(n):    
    ax.scatter(data[:,0][clusters.labels_ == i],data[:,1][clusters.labels_ == i],data[:,2][clusters.labels_ == i],s=0.5)

box.drawBoxes(ax)
axisEqual(data,ax)
plot.xlabel("x")
plot.ylabel("y")

del(box)
