from ast import increment_lineno
import matplotlib
import math
import numpy as np
import seaborn as sns
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from torch import batch_norm_stats
# %matplotlib inline

layers=['Conv1']#,'Conv3','Conv4','Conv6','Conv7']#,'FC1', 'FC2']
# npy_files = ['']
i_image = 24
batchsize = 8
blocksize = 32
xbar_size = 128
i_xbar = 0
input_size = 128

layer = 'Conv1'
input_L1_100IMAGE = np.load('input'+layer+'_batch_size_100_int32.npy')

input_image = input_L1_100IMAGE[i_image]
threshold = 35

class InputVec:
    def __init__(self, index, threshold, acc_blocks):
        self.index = index # record its index to restore the computation result
        self.threshold = threshold # to decide hot or cold
        self.acc_blocks = acc_blocks # accumulated activation numbers of each block in this input vector with 8-bit length 
        self.activated_blocks = np.zeros_like(acc_blocks) # record cold and hot 
        self.var = np.var(acc_blocks)
        self.std = np.std(acc_blocks)
        self.maxindex = np.argmax(acc_blocks)
        self.maxVector = np.zeros_like(acc_blocks)
        self.centroidVec = np.zeros_like(acc_blocks)
        self.maxVector[self.maxindex] = 1
        
    def getVecActivated(self):
        for i_block, accBlock in enumerate(self.acc_blocks):
            if accBlock >= self.threshold:
                self.activated_blocks[i_block] = 1

    def getCentroid(self):
        weightedSum = 0
        weightSum=0
        for i_block, acc_block in enumerate(self.acc_blocks):
            weightedSum += (0.5+i_block)*acc_block
            weightSum += acc_block
        centroid = math.ceil((4*weightedSum/weightSum - 2)/3)
        self.centroidVec[centroid] = 1
inputVectors = []


# for now one time execution
for i, layer in enumerate(layers):
        
    input_L1_100IMAGE = np.load('input'+layer+'_batch_size_100_int32.npy')
    i_image = 24
    batchsizes = [8] # [1,8]
    blocksizes = [32] # [4, 8 ,16, 32]
    xbar_size = 128
    input_image = input_L1_100IMAGE[i_image]
    print("input_image.shape")
    print(input_image.shape)
    sns.set() 
    i_xbar = 0
    plot_size = 128
    input_size = plot_size*8
    start_input = 3200
    block_size = 32
    # batch_size = 8

    for i_plot in range(8):
        input_xbar = input_image[i_xbar*xbar_size:(i_xbar+1)*xbar_size,start_input+plot_size*i_plot:start_input+plot_size*(i_plot+1)]
        
        for i_batchsize, batch_size in enumerate(batchsizes):
            input_blocks_acc = np.zeros((xbar_size//block_size, plot_size//batch_size), dtype=int)
            for j_inPeriod in range(plot_size):
                for i_row in range(xbar_size):
                    i_block = i_row//block_size
                    j_batch = j_inPeriod//batch_size
                    input_blocks_acc[i_block, j_batch] += input_xbar[i_row, j_inPeriod]

            for i_blocks_acc in range(input_blocks_acc.shape[1]): # (i_plot*input_blocks_acc.shape[1], (i_plot+1)*input_blocks_acc.shape[1]):
                inputVectors.append(InputVec(i_blocks_acc, threshold, np.array(input_blocks_acc[:,i_blocks_acc])))
                
    
    for i_inputVecs, inputVec in enumerate(inputVectors):
        inputVec.getVecActivated()
        inputVec.getCentroid()
        print(inputVec.activated_blocks, inputVec.acc_blocks, inputVec.centroidVec, 'var = {}, std = {}'.format(inputVec.var, inputVec.std))
        # print(inputVec.acc_blocks)
            # print(input_blocks_acc.shape)
            # # df = DataFrame.from_records()
            # # index and columns are for plotting
            # index = range(input_xbar.shape[0]//block_size)
            # columns = range(plot_size//batch_size)
            # df = pd.DataFrame(data = input_blocks_acc, columns = columns, index = index)
            # if batch_size == 8:
            #     center = block_size
            # elif batch_size == 1:
            #     center = block_size/4
            # hm, ax = plt.subplots(figsize = (plot_size//batch_size,xbar_size//block_size))
            # hm = sns.heatmap(data = df, square=False, center=center, linewidths=0.1, annot=True, fmt = "d", cmap="RdBu_r") #, center=xbar_size//block_size/4
            # plt.savefig('image_'+layer+'/batch_{}bit/blocksize_{}/cmap_RdBu_r/sample{}_input_image{}_xbar{}_start{}.png'.format(batch_size, block_size, i_plot, i_image, i_xbar, start_input), format = 'png')
            # # plt.savefig('sample_image__RdBu_r.png')
            # hm.clear()
            # plt.close()

            # hm, ax = plt.subplots(figsize = (plot_size//batch_size,xbar_size//block_size))
            # hm = sns.heatmap(data = df, square=False, center=center, linewidths=0.1, annot=True, fmt = "d", cmap="OrRd") #, center=xbar_size//block_size/4
            # plt.savefig('image_'+layer+'/batch_{}bit/blocksize_{}/cmap_OrRd/sample{}_input_image{}_xbar{}_start{}.png'.format(batch_size, block_size, i_plot, i_image, i_xbar, start_input), format = 'png')
            # # plt.savefig('sample_image__OrRd.png')
            # plt.close()
    # for i_batchsize, batch_size in enumerate(batchsizes):
    #     for i_blocksize, block_size in enumerate(blocksizes):
    #         for i_start in range(input_image.shape[1]//input_size):
    #             start_input = input_size*i_start
    #             input_xbar = input_image[i_xbar*xbar_size:(i_xbar+1)*xbar_size,start_input:start_input+input_size]
    #             index = range(input_xbar.shape[0]//block_size)
    #             columns = range(input_size//batch_size)
    #             input_blocks_acc = np.zeros((xbar_size//block_size, input_size//batch_size), dtype=int)
    #             # divide the input into 4 blocks, that block_size = 32, and accumulate the block input
    #             #for j_inPeriod in range(input_xbar.shape[1]):
    #             for j_inPeriod in range(input_size):
    #                 for i_row in range(xbar_size):
    #                     i_block = i_row//block_size
    #                     j_batch = j_inPeriod//batch_size
    #                     input_blocks_acc[i_block, j_batch] += input_xbar[i_row, j_inPeriod]

    #             print(input_blocks_acc.shape)
    #             # df = DataFrame.from_records()
    #             df = pd.DataFrame(data = input_blocks_acc, columns = columns, index = index)
    #             if batch_size == 8:
    #                 center = block_size
    #             elif batch_size == 1:
    #                 center = block_size/4
    #             hm, ax = plt.subplots(figsize = (input_size//batch_size,xbar_size//block_size))
    #             hm = sns.heatmap(data = df, square=False, center=center, linewidths=0.1, annot=True, fmt = "d", cmap="RdBu_r") #, center=xbar_size//block_size/4
    #             plt.savefig('image_'+layer+'/batch_{}bit/blocksize_{}/cmap_RdBu_r/input_image{}_xbar{}_start{}.png'.format(batch_size, block_size, i_image, i_xbar, start_input), format = 'png')
    #             hm.clear()
    #             plt.close()

    #             hm, ax = plt.subplots(figsize = (input_size//batch_size,xbar_size//block_size))
    #             hm = sns.heatmap(data = df, square=False, center=center, linewidths=0.1, annot=True, fmt = "d", cmap="OrRd") #, center=xbar_size//block_size/4
    #             plt.savefig('image_'+layer+'/batch_{}bit/blocksize_{}/cmap_OrRd/input_image{}_xbar{}_start{}.png'.format(batch_size, block_size, i_image, i_xbar, start_input), format = 'png')

    #             # hm.close()
    #             plt.close()
