from ast import increment_lineno
import matplotlib
import numpy as np
import seaborn as sns
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from torch import batch_norm_stats
# %matplotlib inline

layers=['Conv1']#,'Conv3','Conv4','Conv6','Conv7']#,'FC1', 'FC2']
categoryCounts = []
activatedCounts_allXbars = []
activatedCounts_allBatches = []
for i_batch in range(5):
    for i_xbar in range(9): #[0]: 
        activatedCounts_xbar = np.zeros((4)).astype(int)
        activatedCounts_allXbars.append(activatedCounts_xbar)
    activatedCounts_allBatches.append(activatedCounts_allXbars)
    activatedCounts_allXbars=[]
# activatedCounts_sum = np.zeros((4)).astype(int)
print(len(activatedCounts_allBatches),len(activatedCounts_allBatches[0]),len(activatedCounts_allBatches[0][0]))
layer = layers[0]; xbar_rows = 128; batch_size = 8; block_size = 32
input_L1_500IMAGE = np.load('input_0/inputConv1_batch_size=500_FP_int32.npy')
# for i_batch in range(5):
#     input_L1_100IMAGE = input_L1_500IMAGE[100*i_batch:100*(i_batch+1),:,:]
#     for i_image in range(100):
#         input_image = input_L1_100IMAGE[i_image]
#         for i_xbar in range(9): #[0]: 
#             input_xbar = input_image[i_xbar*xbar_rows:(i_xbar+1)*xbar_rows,:]
#             for i_block in range(xbar_rows//block_size):
#                 block_sum = sum(input_xbar[i_block*block_size:(i_block+1)*block_size])
#                 block_sum_arr = np.asarray(block_sum)
#                 print("i_block ", i_block, " ",block_sum_arr.sum(0))
#                 activatedCounts_allBatches[i_batch][i_xbar][i_block]+=block_sum_arr.sum(0)
#     # activatedCounts_allBatches
# with open('input_0/activatedCounts_allBatches.csv','ab+')as f:
#     for i_batch in range(5):
#         np.savetxt(f, activatedCounts_allBatches[i_batch], delimiter=',')
#         f.write(b"\n")
image_block_sum = np.zeros((500,9,5)); images_pes_maxdiff_sort = []
for i_image in range(500):
    input_image = input_L1_500IMAGE[i_image]
    for i_xbar in range(9):
        input_xbar = input_image[i_xbar*xbar_rows:(i_xbar+1)*xbar_rows,:]
        # print("input_xbar.shape, ", input_xbar.shape)
        for i_block in range(xbar_rows//block_size):
            input_block = input_xbar[i_block*block_size:(i_block+1)*block_size]
            # print("input_block.shape, ", input_block.shape)
            block_sum = sum(input_block)
            # print("block_sum.shape, ", block_sum.shape)
            image_block_sum[i_image,i_xbar,i_block] = sum(block_sum)
        
        image_block_sum[i_image,i_xbar,4] = max(image_block_sum[i_image,i_xbar,0:4])-min(image_block_sum[i_image,i_xbar,0:4])

# for i_xbar in range(9):
for i_xbar in [2]:
    images_maxdiff = image_block_sum[:,i_xbar,4]
    image_indexes = np.asarray(range(500))
    images_maxdiff = np.c_[images_maxdiff, image_indexes.T]
    images_maxdiff_sorted = images_maxdiff[np.argsort(images_maxdiff[:,0])]
    # images_pes_maxdiff_sort.append(images_maxdiff_sorted)
    print(images_maxdiff_sorted[0:30,1].T, "\n")

# with open('input_0/images_pes_maxdiff_sort.csv','ab+')as f:
#     np.savetxt(f, images_pes_maxdiff_sort[i_batch], delimiter=',')
