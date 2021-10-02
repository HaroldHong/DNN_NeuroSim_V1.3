# This is for setting input file with threshold number of 1 and transport back to NeuroSim

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
# npy_files = ['']
threshold = 48
# input_l1_100images = np.load('input'+layers[0]+'_batch_size_100_int32.npy')
batchsizes = [8]#1,8]
blocksizes = [32]# 4, 8 ,16, 32]
xbar_size = 128
batch_size = batchsizes[0]
block_size = blocksizes[0]
for i, layer in enumerate(layers):
        
    input_L1_100IMAGE = np.load('input'+layer+'_batch_size_100_int32.npy')
    input_L1_100IMAGE_insert = np.zeros_like(input_L1_100IMAGE)
    for i_image in range(input_L1_100IMAGE.shape[0]):
    # i_image = 24
   
        input_image = input_L1_100IMAGE[i_image]
        input_image_insert = np.zeros_like(input_image)
        # print("input_image.shape")
        # print(input_image.shape)
        sns.set() 
        # i_xbar = 0
        input_size = 128
        for i_xbar in range(input_image.shape[0]//xbar_size):
            for i_start in range(input_image.shape[1]//input_size):
                start_input = input_size*i_start
                input_xbar = input_image[i_xbar*xbar_size:(i_xbar+1)*xbar_size,start_input:start_input+input_size]
                input_xbar_insert = np.zeros_like(input_xbar)
                for i_batch in range(xbar_size//batch_size):
                    for i_one in range(threshold):
                        # print('i_batch*batch_size+i_one\%block_size = {}'.format(i_batch*batch_size+i_one%block_size))
                        input_xbar_insert[i_one%block_size,i_batch*batch_size+i_one//block_size] = 1
                input_image_insert[i_xbar*xbar_size:(i_xbar+1)*xbar_size,start_input:start_input+input_size] = input_xbar_insert
                
                input_blocks_acc = np.zeros((xbar_size//block_size, input_size//batch_size), dtype=int)
                
                # divide the input into 4 blocks, that block_size = 32, and accumulate the block input
                for j_inPeriod in range(input_size):
                    for i_row in range(xbar_size):
                        i_block = i_row//block_size
                        j_batch = j_inPeriod//batch_size
                        input_blocks_acc[i_block, j_batch] += input_xbar_insert[i_row, j_inPeriod]
                # print(input_blocks_acc)

        input_L1_100IMAGE_insert[i_image] = input_image_insert

    savefile = 'input'+layer+'_batch_size_100_int32_threshold{}.npy'.format(threshold)
    np.save(savefile, input_L1_100IMAGE_insert)
        # index = range(input_xbar.shape[0]//block_size)
        # columns = range(input_size//batch_size)
        # df = DataFrame.from_records()
        # df = pd.DataFrame(data = input_blocks_acc, columns = columns, index = index)
        # if batch_size == 8:
        #     center = block_size
        # elif batch_size == 1:
        #     center = block_size/4
        # hm, ax = plt.subplots(figsize = (input_size//batch_size,xbar_size//block_size))
        # hm = sns.heatmap(data = df, square=False, center=center, linewidths=0.1, annot=True, fmt = "d", cmap="RdBu_r") #, center=xbar_size//block_size/4
        # plt.savefig('image_'+layer+'/batch_{}bit/blocksize_{}/cmap_RdBu_r/input_image{}_xbar{}_start{}.png'.format(batch_size, block_size, i_image, i_xbar, start_input), format = 'png')
        # hm.clear()
        # plt.close()

        # hm, ax = plt.subplots(figsize = (input_size//batch_size,xbar_size//block_size))
        # hm = sns.heatmap(data = df, square=False, center=center, linewidths=0.1, annot=True, fmt = "d", cmap="OrRd") #, center=xbar_size//block_size/4
        # plt.savefig('image_'+layer+'/batch_{}bit/blocksize_{}/cmap_OrRd/input_image{}_xbar{}_start{}.png'.format(batch_size, block_size, i_image, i_xbar, start_input), format = 'png')

        # # hm.close()
        # plt.close()