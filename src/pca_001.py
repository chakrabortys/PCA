import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from numpy import linalg as lng
import math

def readimage():
    pic = Image.open("{Image_Path}.jpg")
    r,g,b = pic.split()
    red_data = np.array(r.getdata())
    r_ary = red_data.reshape(1713,3448) # Image dimension
    random_patch(r_ary)

def random_patch(ary):
    samp_num = 1000
    ptch_sz = 16
    correlation = np.zeros((256,256))
    row_start = np.random.choice(ary.shape[0] - ptch_sz - 1, size=samp_num, replace=False)
    col_start = np.random.choice(ary.shape[1]- ptch_sz - 1, size=samp_num, replace=False)
    for n in range(samp_num):
        r_in = row_start[n]
        c_in = col_start[n]
        samp = ary[r_in:r_in+ptch_sz,c_in:c_in+ptch_sz]
        product = correlation_mtx(samp)
        correlation = correlation + product
    eigenvectors(correlation)

def correlation_mtx (samp) :
    v_length = samp.shape[0]*samp.shape[1]
    col_v = np.ravel(samp).reshape(v_length,1)
    row_v = np.transpose(col_v)
    prod = np.matmul(col_v,row_v)
    return prod

def eigenvectors(correlation):
    #pca = Image.new('RGB', (300, 300))
    w, v = lng.eig(correlation)
    sort_idx = w.argsort()[::-1]
    #print(sort_idx)
    for j in range(64) :
        eig_top = v[:,sort_idx[j]]
        print(eig_top)
        patch = eig_top.reshape(16,16)
        #im = Image.fromarray(patch)
        #pca = show_img(im,pca,j)
        mpimg.imsave("{Path}/sample%20d.png"%j, patch, cmap='gray')
show_img()

def show_img() :
    pca = Image.new('RGB',(150,150))
    for i in range(64):
        im = Image.open("{Path}/sample%20d.png" % i)
        im.thumbnail((16,16),Image.ANTIALIAS)
        y = (math.floor(i / 8)*16) +2*math.floor(i / 8)
        x = ((i % 8) * 16 )+2*(i%8)
        #print(x,y)
        pca.paste(im, (x, y, x + 16, y + 16))
    pca.save("{Path}/sample.png")

readimage()





