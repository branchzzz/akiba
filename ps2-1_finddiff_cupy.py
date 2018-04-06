#!/usr/bin/env python
from __future__ import print_function, division
import os
import numpy as np
import scipy.misc
import argparse
import sklearn.cluster
from chainer import cuda

def loadimage(path):
    return np.array(scipy.misc.imread(path),dtype=np.float32)/255.0


def to_cpu(x):
    if isinstance(x, cuda.cupy.core.core.ndarray):
        return cuda.to_cpu(x)
    return x

def saveimage(path,img):
    img = (to_cpu(img)*255.0).astype(np.uint8)
    return scipy.misc.imsave(path,img)


def gaussian_nll(x,mean,std, minstd=0.1,xp=np):
    var = xp.clip(std,minstd,1.0)**2
    negative_loglikelihood = xp.log(xp.sqrt(2*np.pi*var))+(x-mean)**2/(2*var)
    return negative_loglikelihood


def main():
    parser = argparse.ArgumentParser()
    # parameter
    parser.add_argument("--gpu",    type=int,   default=-1,
                         help="if you want to use GPU, setting --gpu more than 0")
    parser.add_argument("--folder",      type=str,   default="/home/zhouboqian/akiba/finddiff/dataset/",
                        help="folder containing movies")
    parser.add_argument("--movieid",    type=int,   default=0,
                        help="id of movie")
    parser.add_argument("--n_ave",       type=int,   default=50,
                        help="number of frames to calculate background")
    parser.add_argument("--fore_th",     type=float, default=20.0,
                        help="threshold of nll to consider as foreground")
    parser.add_argument("--minstd",      type=float, default=1e-6,
                        help="clipping std of background")
    parser.add_argument("--detectcolor", type=str,   default="[1.0,0.0,0.0]",
                        help="color to denote foreground")
    parser.add_argument("--bboxcolor", type=str,   default="[0.0,0.0,1.0]",
                        help="color to denote bbox")
    parser.add_argument("--n_clusters",  type=int,   default=200,
                        help="number of clusters of k-means, i.e., maximum number of humans")
    parser.add_argument("--cluster_dist",type=float, default=15,
                        help="threshold of distance to merge clusters")
    parser.add_argument("--cluster_th",  type=float, default=10,
                        help="threshold of points in clusters between human and noise")
    parser.add_argument("--detectsize",  type=int,   default=5,
                        help="size of square to denote detected humans")
    # target
    parser.add_argument("--output",      type=str, default="cluster",
                        help="output data")
    # output
    parser.add_argument("--log",   type=str, default=None)
    opt = parser.parse_args()

    if opt.log is None:
        opt.log="results/{:05d}/".format(opt.movieid)
    else:
        opt.log=opt.log+"{:05d}/".format(opt.movieid)
    opt.folder=opt.folder+"{:05d}/".format(opt.movieid)
    print(opt)
    
    # import ipdb
    # ipdb.set_trace()
    # make output folder
    try:
        os.makedirs(opt.log)
    except OSError:
        pass

    # cupy
    if opt.gpu >= 0:
        cuda.get_device(opt.gpu).use()
        xp = cuda.cupy
    else:
        xp = np

    # load files
    filenames_all = sorted(os.listdir(opt.folder))
    #index_interested = opt.n_ave//2# 
    index_interested = 2
    opt.detectcolor = np.array(eval(opt.detectcolor))[np.newaxis,np.newaxis,:]
    opt.bboxcolor = xp.array(eval(opt.bboxcolor))
    filenames = []
    images_loaded = []
    images_normed = []
    last_n_clusters=opt.n_clusters
    for itr in range(len(filenames_all)):
        print("load file: {}".format(filenames_all[itr]))
        filenames.append(filenames_all[itr])
        image_loaded=xp.array(loadimage(opt.folder+filenames_all[itr]))
        image_normed=(image_loaded-image_loaded.mean(axis=(0,1)))/image_loaded.std(axis=(0,1))
        image_normed=xp.expand_dims(image_normed,axis=0)
        images_loaded.append(image_loaded)
        images_normed.append(image_normed)
        #if len(images_normed) < opt.n_ave+1:#
        if len(images_normed) < len(filenames_all)+1:
            continue
        print("background subtraction: {}".format(filenames[index_interested]))
        image_loaded = images_loaded[index_interested]
        image_normed = images_normed[index_interested]
        filename = filenames[index_interested]
        images_pre_post = xp.vstack(images_normed[:index_interested]+images_normed[index_interested+1:])
        mean = images_pre_post.mean(axis=0)
        std = images_pre_post.std(axis=0)
        nll = gaussian_nll(image_normed[0],mean,std,opt.minstd,xp=xp).sum(axis=2)
        image_foreground = (nll > opt.fore_th).astype(np.float32)[:,:,np.newaxis]
        # clustering detected areas
        print("process: cluster")
        foreground_points = np.array(np.where(to_cpu(image_foreground[:,:,0])))
        n_clusters=min(last_n_clusters+10,opt.n_clusters)
        kminit="k-means++"
        while True:
            km=sklearn.cluster.KMeans(n_clusters=n_clusters, init=kminit)
            km.fit(foreground_points.T)
            # number of dots in clusters
            label_counts = np.bincount(km.labels_)
            # distance between clusters
            centers = km.cluster_centers_
            c_distance=np.sqrt(((np.expand_dims(centers,axis=1)-np.expand_dims(centers,axis=0))**2).sum(axis=2))
            near_clusters=np.array(np.where(np.logical_and(c_distance<opt.cluster_dist,c_distance>0)))
            if near_clusters.size==0:
                break
            kminit = centers.copy()
            # remove smallest cluster and use as initial cluster for next fitting
            kminit=np.delete(kminit, near_clusters[0][np.argmin(label_counts[near_clusters[0]])], axis=0)
            # remove the clusters located near other clusters
            # kminit=np.delete(kminit, near_clusters[0][:near_clusters.shape[1]//2], axis=0)
            n_clusters=kminit.shape[0]
            print("process: cluster, near->",near_clusters[0],"/",n_clusters)
        last_n_clusters=n_clusters
        # use clusters having many dots
        centers=centers[label_counts>=opt.cluster_th]

        print("process: cluster, outputting file")
        centers=(centers+0.5).astype(int)
        bbox=np.hstack([centers-opt.detectsize,centers+opt.detectsize])
        np.savetxt(opt.log+"bbox_"+filename[:-filename[::-1].index(".")-1]+".dat",bbox, fmt="%d")

        print("process: cluster, making image")
        image_detected = (1-image_foreground)*image_loaded+image_foreground*opt.bboxcolor
        image_detected = to_cpu(image_detected)
        detectsize = np.arange(-opt.detectsize,opt.detectsize+1)
        detectsize = np.tile(detectsize,(opt.detectsize*2+1,1))
        cx,cy = centers.T[:,:,np.newaxis,np.newaxis]
        cx = (cx+detectsize).reshape(-1)
        cy = (cy+detectsize.T).reshape(-1)
        in_area=np.logical_and(np.logical_and(
            cx<image_detected.shape[0],
            cx>=0
        ),
        np.logical_and(
            cy<image_detected.shape[1],
            cy>=0
        ))
        cx=cx[in_area]
        cy=cy[in_area]
        image_detected[cx,cy,:] = opt.detectcolor
        saveimage(opt.log+"centered_"+filename,image_detected)
        saveimage(opt.log+"original_"+filename,image_loaded)

        with open(opt.log+"info_"+filename[:-filename[::-1].index(".")-1]+".dat","w") as of:
            print("#",opt,file=of)
            print("# foreground_points",foreground_points.shape[1],sep="\n",file=of)

            # import ipdb
            # ipdb.set_trace()
        filenames = filenames[1:]
        images_loaded = images_loaded[1:]
        images_normed = images_normed[1:]

if __name__ == '__main__':
    main()
