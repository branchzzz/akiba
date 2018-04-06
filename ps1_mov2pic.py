#!/usr/bin/python
# coding: UTF-8
import sys
sys.path.append('/home/yamauchi/workspace/my_module')
#import post_to_slack

import os
import subprocess
# import ipdb
from time import time

def batch_finddiff():
    start=time()
    procs=[]
    os.chdir("/home/yamauchi/workspace/mot/akiba/matsubara_code/")
    print os.getcwd()
    for idx,i in enumerate(range(0,39)):
        number_padded = str("{0:05d}".format(i))
        gpu = str((i%3)+3)
        data_dir = "/home/share/akiba/12fps/dataset/"
        save_dir = "/home/share/akiba/12fps/results/"

        # cmd = "python finddiff_cupy.py --folder "+data_dir+" --log "+save_dir+" --movieid "+number_padded
        cmd = "python finddiff_cupy.py --gpu "+gpu+" --folder "+data_dir+" --log "+save_dir+" --movieid "+number_padded
        print cmd
        post_to_slack.post(cmd+"start")

        # os.system(cmd)
        proc = subprocess.Popen(cmd.split())
        procs.append(proc)

        if (idx+1)%3 == 0:
            for proc in procs:
                proc.communicate()
                # post_to_slack.post(str(idx)+"finish")
    
    post_to_slack.post("finish")


    end=time()
    print("%f sec" %(end-start))

def mkdir():
    for idx,i in enumerate(range(0,39)):
        number_padded = str("{0:05d}".format(i))
        cmd = "mkdir /home/zhouboqian/akiba/12fps/"+number_padded
        print cmd
        os.system(cmd)

def main():
    start=time()
    procs=[]
    os.chdir("/home/yamauchi/workspace/akiba/data/20170725denkigaiguchi/")
    print os.getcwd()
    for idx,i in enumerate(range(25,39)):
        number_padded = str("{0:05d}".format(i))
        cmd = "mv "+number_padded+".MTS /data2/yamauchi/20170725denkigaiguchi/"
        print cmd
        os.system(cmd)

def mov2pic():
    start=time()
    procs=[]
    os.chdir("/home/zhouboqian/akiba/12fps/")
    print os.getcwd()
    for idx,i in enumerate(range(0,39)):
        number_padded = str("{0:05d}".format(i))
        data_dir = "/data2/yamauchi/akiba/12fps/data/"+number_padded+".MTS"
        save_dir = "/home/zhouboqian/akiba/12fps/"+number_padded
        # os.system("mkdir "+number_padded)
        cmd = "ffmpeg -i "+data_dir+" -framerate 12 -f image2 -vcodec mjpeg -qscale 1 -qmin 1 -qmax 1 "+save_dir+"/frame%08d.jpeg"
        print cmd
        # os.system(cmd)
        proc = subprocess.Popen(cmd.split())
        procs.append(proc)

        if (idx+1)%13 == 0:
            for proc in procs:
                proc.communicate()

    end=time()
    print("%f sec" %(end-start))

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--holder",  type=str,   default="00001")
    # parser.add_argument("--maxdist", type=int, default=30)
    # args = parser.parse_args()
    
    # batch_finddiff()
    mov2pic()
