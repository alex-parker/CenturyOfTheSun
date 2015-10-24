###--- Century of the Sun

import numpy as np
import numpy
import matplotlib.pyplot as plt
import sys
from numpy.linalg import eig, inv
from skimage import data, color
from skimage.transform import hough_circle
from skimage.feature import peak_local_max, canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte

from PIL import Image

from skimage.filters import threshold_otsu
from sklearn.cluster import AffinityPropagation, DBSCAN
from scipy.signal import convolve


def center_box(im, center, box):
	center = np.rint(center)
	box = np.rint(box)
	print [center[0]-box[0], center[0]+box[0], center[1]-box[1], center[1]+box[1]]
	return im[center[0]-box[0]:center[0]+box[0], center[1]-box[1]:center[1]+box[1]]


def fitEllipse(x,y):
    x = x[:,np.newaxis]
    y = y[:,np.newaxis]
    D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    S = np.dot(D.T,D)
    C = np.zeros([6,6])
    C[0,2] = C[2,0] = 2; C[1,1] = -1
    E, V =  eig(np.dot(inv(S), C))
    n = np.argmax(np.abs(E))
    a = V[:,n]
    return a	

def ellipse_center(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    num = b*b-a*c
    x0=(c*d-b*f)/num
    y0=(a*f-b*d)/num
    return np.array([x0,y0])


def ellipse_angle_of_rotation( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    return 0.5*np.arctan(2*b/(a-c))


def ellipse_axis_length( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    res1=np.sqrt(up/down1)
    res2=np.sqrt(up/down2)
    return np.array([res1, res2])

def highpass(im):

	k = numpy.zeros((7,7))-1
	k[3,3] = 56
	return convolve(im, k, mode='same')

def run(fn, ax):
	# Load picture and detect edges
	ax.cla()
	image = img_as_ubyte(Image.open(fn))

	r_max = 0.5 * max(image.shape)

	image = np.pad(image, 500, mode='constant', constant_values=255)

	#fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(16, 8))
	#ax.imshow(image, cmap=plt.cm.gray)
	#plt.show()

	print image.max()

	thresh = threshold_otsu(image)
	edges = highpass(image) > thresh

	inds = np.indices(image.shape) #- 0.5 * np.array(image.shape)
	y,x = inds[0] - 0.5 * image.shape[0], inds[1] - 0.5 * image.shape[1]

	r = (x**2 + y**2)**0.5

	mask = r < r_max - 25

	edges[mask] = True

	inds = numpy.indices(edges.shape)

	edges = np.logical_not(edges)

	x, y = inds[0].ravel()[edges.ravel()], inds[1].ravel()[edges.ravel()]



	print 'Clustering'
	X = numpy.asarray([x,y]).T
	db = DBSCAN(eps=5, min_samples=10).fit(X)
	core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
	core_samples_mask[db.core_sample_indices_] = True
	labels = db.labels_
	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

	print 'Clusters: ', n_clusters_

	unique_labels = np.unique(labels)
	nv = []
	for label in unique_labels:
		n = numpy.sum(labels==label)
		nv.append(n)

	inds = unique_labels[ np.argsort(nv)[-2:] ]

	print inds

	x, y = np.append(x[labels==inds[0]], x[labels==inds[1]]), np.append(y[labels==inds[0]], y[labels==inds[1]])

	#fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(16, 9))

	ell = fitEllipse(x,y)

	center =ellipse_center(ell)
	phi = ellipse_angle_of_rotation(ell)
	axes = ellipse_axis_length(ell)

	print("center = ",  center[0], center[1])
	print("angle of rotation = ",  180.0 * phi / np.pi)
	print("axes = ", axes[0], axes[1])

	R = numpy.linspace(0, 2.0 * np.pi, 1000)

	a = axes[0]
	b = axes[1]

	eqsd = ( 0.5 * (a**2 + b**2) )**0.5 + 200.0

	xx = center[0] + a*np.cos(R)*np.cos(phi) - b*np.sin(R)*np.sin(phi)
	yy = center[1] + a*np.cos(R)*np.sin(phi) + b*np.sin(R)*np.cos(phi)

	#print 
	im = center_box(image, center, [9.0*eqsd/16.0, eqsd ])

	ax.imshow(im, cmap=plt.cm.gray)
	#ax.scatter(yy, xx, facecolor='b', edgecolor='none', marker='.')

	ax.set_xticks([])
	ax.set_yticks([])
	plt.draw()
	plt.savefig('centered_%s.png'%(fn))

plt.ion()
fig = plt.figure( figsize=(16, 9))
ax = fig.add_axes([0,0,1,1])

for fn in sys.argv[1:]:
	run(fn, ax)