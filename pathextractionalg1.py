"""
What is NumPy?
NumPy is the fundamental package for scientific computing in Python.
 It is a Python library that provides a multidimensional array object,
 various derived objects (such as masked arrays and matrices), 
 and an assortment of routines for fast operations on arrays,
 including mathematical, logical, shape manipulation, sorting, 
 selecting, I/O, discrete Fourier transforms, basic linear algebra,
 basic statistical operations, random simulation and much more
"""

"""
Matplotlib is a comprehensive library for creating static,
 animated, and interactive visualizations in Python.
"""


"""
SciPy (pronounced “Sigh Pie”) is a Python-based ecosystem of open-source software for
 mathematics, science, and engineering
"""

"""
scikit-image is an image processing toolbox for SciPy
"""

"""
NetworkX is a Python package for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks.



"""

import numpy as np
from matplotlib import pyplot as plt
from numpy import invert ,pi
from scipy import ndimage
import skimage
from skimage.morphology import skeletonize
from skimage.io import imread,imshow
from skimage.filters import threshold_otsu
from skimage.measure import regionprops, label
import networkx as nx
from matplotlib import collections as mc
from matplotlib.animation import FuncAnimation
import matplotlib.animation as ani
import random
from random import random
from scipy.ndimage import gaussian_filter
from time import sleep
import sys


#read the the image and rescale it:
########################################################
#img=imread('gwin.png')

img=imread("image1.jpg")
T=1 # 1 for plotting image else 0 




try:
 
 img=skimage.color.rgb2gray(img)
 
except:
 print("An exception occurred")
    









res=300 #resolution of the image 
res1=res

sig=2 # used to smooth the signal and filter noise from the path 


print(img.shape)
a,b=img.shape


res=res/((a*b)**0.5)#the new number of pixels = x**2  
img=skimage.transform.rescale(img, res, anti_aliasing=True)
img1=img

a,b=img.shape
#######################################################



sizeim=img.shape
print(sizeim) 

#thresholing the the image and skeletonize it:
#######################################################
#2img=skimage.filters.gaussian(img,sigma=1)

thresh=threshold_otsu(img) #threshold value of the image
binary_img=img >thresh
binary_img=invert(binary_img) #threshold th image





#remove noise and small object
#######################################################
#generate multiple objects we select the biggest object 

#binary_img= ndimage.median_filter(binary_img, size=5)#remove noise
labels,labelnum=skimage.measure.label(binary_img, background=None, return_num=True, connectivity=None)
hist,bin_edges=np.histogram(labels,bins=labelnum+1)


sorted_hist=np.sort(hist)
print(hist,sorted_hist[-1])
#maxi=np.where(hist==max(hist[1:]))
maxi=np.where(hist==sorted_hist[-2])
maxi=maxi[0]

print("maxi= ",maxi)

binary_img=np.where(labels==maxi,1,0)  #select  the biggest object 
#######################################################222
plt.matshow(labels)
if T: plt.show()



skel_img=skeletonize( binary_img ) #skeletonize the image






plt.matshow(skel_img)
ar=np.where(skel_img,1,0) # convert the skeletonised image to numpy array
 
 
 



if T: plt.show()



#######################################################


fig,(ax1,ax2)=plt.subplots(1,2)
fig.suptitle('Horizontally stacked subplots')
ax1.matshow( img)








# register the points representing the skeletonised image in the P array , P from points
#######################################################

x,y=np.where(ar==0)
P=np.stack((x,y),axis=1)
P=tuple(map(tuple,P.astype(np.int16)))
# p contain points coordinate value that repesent the skeletonised image
#######################################################



#GRAPH__________________________
#######################################################
import time

t0 = time.time()

G = nx.grid_2d_graph(a,b)

t_add_edge0=time.time()
G.add_edges_from([((x, y), (x+1, y+1)) for x in range(a-1) for y in range(b-1)]
 + [((x+1, y), (x, y+1))for x in range(a-1) for y in range(b-1)])






G.remove_nodes_from(P)

MG=nx.MultiGraph(G)

MG.add_edges_from(G.edges) #double every edge in the graph so it will be euler graph

#MG is the graph representation of the skeletonised image where each pixel transform to node 
#generate links between the nieberhood pixels

# we can imagine the skeletonised image as big city where node is an intersection and link is a road

t_add_edge1=time.time() # used to compute the time for convert the image to graph and eulerize it
print("time creating graph= ",t_add_edge1-t_add_edge0)
tf=nx.is_eulerian(MG) 

print(tf)


#G=nx.eulerize(G)
path=nx.eulerian_circuit(MG)
#######################################################

t1 = time.time()



total = t1-t0


print("totla=",total)

pos = {(x,y):(y,-x) for x,y in G.nodes()}
nx.draw(MG,node_size=20,pos=pos,with_labels=False) # print the graph


if T: plt.show()

    
    
    


#create two signals X Y from the path:
#######################################################
path=list(path)
points=[]
for i in path:
    points.append(i[0])

#print(path)
#print(points)

X,Y=zip(*points)
X=list(X)
Y=list(Y)



#X=X[10:-10]
#Y=Y[10:-10]
X=gaussian_filter(X,sigma=sig ,mode='reflect')  # used to remove the noise 
Y=gaussian_filter(Y,sigma=sig ,mode='reflect')


#print(NewXY.shape)
#######################################################


#animation part:
#######################################################
fig, axs = plt.subplots(2,2)


axs[0,0].plot(X,color='r')
axs[0,0].set_xlabel('time')
axs[0,0].set_ylabel('xmotposition')
#axs[0,0].plot(Y,color='g')
axs[0,1].plot(Y,color='g')
axs[0,1].set_xlabel('time')
axs[0,1].set_ylabel('ymotposition')


Xd=np.gradient(X) 
Xd=gaussian_filter(Xd, sigma=3)
Yd=np.gradient(Y)

Yd=gaussian_filter(Yd, sigma=3)


axs[1,0].plot(Xd,color='r')
axs[1,0].set_xlabel('time')
axs[1,1].set_ylabel('xspeed')
axs[1,1].plot(Yd,color='g')
axs[1,1].plot(Yd,color='g')
axs[1,0].set_xlabel('time')
axs[1,1].set_ylabel('yspeed')


fig=plt.figure()

plt.ylim(a, 0)
plt.xlim(0,b)




po,=plt.plot(Y[0],X[0],'ro',ms=10,mfc='r')
po1,=plt.plot(Y[0],X[0],'ro',ms=10,mfc='b')




imshow(img1)

y=np.zeros(1)
x=np.zeros(1)
sp=res1/25

def anim(i):
   
    
    global y
    global x
    
    ii=sp*i
  
    
    y=np.append(y,Y[int(ii-sp):int(ii)])
    x=np.append(x,X[int(ii-sp):int(ii)])
    
   
    po.set_data(y,x)
    po1.set_data(Y[int(ii)],X[int(ii)])

    
    
    return po, po1,
    





Animation=FuncAnimation(fig,anim,interval=.1,blit=True,frames=int(len(X)/sp),repeat=False)

print("done")

if T: plt.show()

#######################################################







