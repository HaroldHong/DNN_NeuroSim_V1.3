from ast import increment_lineno
import matplotlib
import math
import sys
import numpy as np
import seaborn as sns
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from torch import batch_norm_stats
# %matplotlib inline

class InputVec:
    def __init__(self, index, threshold, acc_blocks):
        self.index = index # record its index to restore the computation result
        self.threshold = threshold # to decide hot or cold
        self.acc_blocks = acc_blocks # accumulated activation numbers of each block in this input vector with 8-bit length 
        self.activated_blocks = np.zeros_like(acc_blocks).astype('bool') # record cold and hot 
        self.var = np.var(acc_blocks)
        self.std = np.std(acc_blocks)
        self.maxindex = np.argmax(acc_blocks)
        self.maxVector = np.zeros_like(acc_blocks).astype('bool')
        self.centroidVec = np.zeros_like(acc_blocks).astype('bool')
        self.maxVector[self.maxindex] = 1
        self.centroid = 0
        # self.maxBin = pow(2,(self.maxindex - 1))
        
    def getVecActivated(self):
        for i_block, accBlock in enumerate(self.acc_blocks):
            if accBlock >= self.threshold:
                self.activated_blocks[i_block] = 1
                # self.activated_blockBin += pow(2,(i_block - 1))

    def getCentroid(self):
        weightedSum = 0
        weightSum=0
        for i_block, acc_block in enumerate(self.acc_blocks):
            weightedSum += (0.5+i_block)*acc_block
            weightSum += acc_block
        self.centroid = math.floor((4*weightedSum/weightSum - 2)/3)
        self.centroidVec[self.centroid] = 1

    # def getNoActived_blocks(self):
    #     for i_block, activated_block in enumerate(self.activated_blocks):
    #         if(activated_block == 1):
layers=['Conv1']#,'Conv3','Conv4','Conv6','Conv7']#,'FC1', 'FC2']
batchsize = 8
blocksize = 32
xbar_size = 128
i_xbar = 0
# input_size = 128

layer = 'Conv1'
input_L1_100IMAGE = np.load('input'+layer+'_batch_size_100_int32.npy')


categoryCounts = []
activatedCounts = []
activatedCounts_sum = np.zeros((4)).astype(int)
categoryCounts_sum = np.zeros((4)).astype(int)
# for now one time execution
image1 = 86; image2 = 93
input_image1 = input_L1_100IMAGE[image1]
input_image2 = input_L1_100IMAGE[image2]
threshold = 40


for i_image in range(100):
    # for i_xbar in range(100):
    for i, layer in enumerate(layers):
            
        # input_L1_100IMAGE = np.load('input'+layer+'_batch_size_100_int32.npy')
        # i_image = 13
        batchsizes = [8] # [1,8]
        blocksizes = [32] # [4, 8 ,16, 32]
        xbar_size = 128
        input_image = input_L1_100IMAGE[i_image]
        print("input_image.shape")
        print(input_image.shape)
        sns.set() 
        i_xbar = 0
        plot_size = 128
        input_size = plot_size*8*7
        start_input = 0
        block_size = 32
        positiveSeq = [0,1,2,3]
        invertedSeq = [3,2,1,0]
        batch_size = 8
        inputVectors = []
        inputMax0 = []; inputMax1 = []; inputMax2 = []; inputMax3 = []
        inputCategory = [inputMax0, inputMax1, inputMax2, inputMax3]
        # inputCategory = np.asarray(inputCategory)
        inputRearrange = []

        for i_plot in range(input_size//plot_size):
            
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
                # print('inputVectors.size:{}'.format(len(inputVectors)))
        activatedCount = np.zeros((4)).astype(int)
        categoryCount = np.zeros((4)).astype(int)
        for i_inputVecs, inputVec in enumerate(inputVectors):
            inputVec.getVecActivated()
            inputVec.getCentroid()
            inputCategory[inputVec.maxindex].append(inputVec)
            activatedCount += inputVec.activated_blocks.astype(int)
            # print(inputVec.activated_blocks, inputVec.acc_blocks, inputVec.maxVector, 'var = {}, std = {}'.format(inputVec.var, inputVec.std))

        # print(np.array(inputCategory, dtype = object).shape)
        categoryCount = [len(inputCategory[0]), len(inputCategory[1]), len(inputCategory[2]), len(inputCategory[3])]

        activatedCounts_sum += activatedCount
        categoryCounts_sum += categoryCount
        activatedCounts.append(activatedCount)
        categoryCounts.append(categoryCount)
# print(categoryCounts_sum, activatedCounts_sum)
# np.savetxt('./categoryCounts_ori.csv', categoryCounts, delimiter=',')
# np.savetxt('./activatedCounts_ori.csv', activatedCounts, delimiter=',')
        print(len(inputCategory[0]), len(inputCategory[1]), len(inputCategory[2]), len(inputCategory[3]), activatedCount)
        firstVec = inputVectors[0]
        if(inputCategory[firstVec.maxindex][0].index == firstVec.index):
            inputCategory[firstVec.maxindex].pop(0)
        else:
            print("First Vector ERROR!")
            sys.exit(0)
        inputRearrange.append(firstVec)
        for i_no, no_activatedblock in enumerate(~inputRearrange[0].activated_blocks):
            if no_activatedblock == 1 and len(inputCategory[i_no]) != 0:
                nextVector = inputCategory[i_no].pop(0)
                inputRearrange.append(nextVector)
                break
            elif i_no == 3: # first vector is [1111]
                for i_category in range(len(inputCategory)):
                    if(len(inputCategory) != 0):
                        nextVector = inputCategory[i_category].pop(0)
                        inputRearrange.append(nextVector)
                        break
        inputCategory = np.asarray(inputCategory, dtype = 'object')
        while len(inputCategory[0]) != 0 or len(inputCategory[1]) != 0 or len(inputCategory[2]) != 0 or len(inputCategory[3]) != 0:
            no_cur_or_his = ~(inputRearrange[-1].activated_blocks | inputRearrange[-2].maxVector)
            
            # 1. determine no_cur_or_his is not empty
            # np.max(no_cur_or_his) == 1

            # 2. determine whether blocks in no_cur_or_his is null
            candidate_blocks = inputCategory[no_cur_or_his]
            is_candidates_valid = any(candidate_blocks)
            
            # 3. determine traverse order of category blocks 
            # (start from the one far away from the current maxindex)
            if inputRearrange[-1].maxindex > 1:
                traverseOrder = positiveSeq.copy()
            else:
                traverseOrder = invertedSeq.copy()

            traverseOrder.remove(inputRearrange[-1].maxindex)
            traverseOrder.append(inputRearrange[-1].maxindex)
            
            # 4. if is_candidates_valid is True, eliminate other blocks in traverseOrder
            if is_candidates_valid:
                eliminateds = np.argwhere(no_cur_or_his==0)[:,0]
                for eliminated in eliminateds:
                    traverseOrder.remove(eliminated)
            for traverse_block in traverseOrder:
                if len(inputCategory[traverse_block]) != 0:
                    nextVector = inputCategory[traverse_block].pop(0)
                    inputRearrange.append(nextVector)
                    break
        # reconstruct the rearranged input blocks accumulation data
        new_input_blocks_acc = np.zeros((xbar_size//block_size, input_size//batch_size), dtype=int)
        for i_newinput, newinputVec in enumerate(inputRearrange):
            # print(new_input_blocks_acc.shape)
            print(newinputVec.acc_blocks, newinputVec.activated_blocks, newinputVec.maxVector)
            new_input_blocks_acc[:,i_newinput] = newinputVec.acc_blocks

    # print(inputVec.acc_blocks)
    # print(input_blocks_acc.shape)
    # # df = DataFrame.from_records()
    # # index and columns are for plotting


    # index = range(input_xbar.shape[0]//block_size)
    # columns = range(plot_size//batch_size)
    # for i_plot in range(8):
    #     df = pd.DataFrame(data = new_input_blocks_acc[:,i_plot*(plot_size//batch_size):(i_plot+1)*(plot_size//batch_size)], columns = columns, index = index)
    #     if batch_size == 8:
    #         center = block_size
    #     elif batch_size == 1:
    #         center = block_size/4
    #     hm, ax = plt.subplots(figsize = (plot_size//batch_size,xbar_size//block_size))
    #     hm = sns.heatmap(data = df, square=False, center=center, linewidths=0.1, annot=True, fmt = "d", cmap="RdBu_r") #, center=xbar_size//block_size/4
    #     plt.savefig('image_'+layer+'/batch_{}bit/blocksize_{}/cmap_RdBu_r/rearrange{}_input_image{}_xbar{}_start{}.png'.format(batch_size, block_size, i_plot, i_image, i_xbar, start_input), format = 'png')
    #     # plt.savefig('sample_image__RdBu_r.png')
    #     hm.clear()
    #     plt.close()

    #     hm, ax = plt.subplots(figsize = (plot_size//batch_size,xbar_size//block_size))
    #     hm = sns.heatmap(data = df, square=False, center=center, linewidths=0.1, annot=True, fmt = "d", cmap="OrRd") #, center=xbar_size//block_size/4
    #     plt.savefig('image_'+layer+'/batch_{}bit/blocksize_{}/cmap_OrRd/rearrange{}_input_image{}_xbar{}_start{}.png'.format(batch_size, block_size, i_plot, i_image, i_xbar, start_input), format = 'png')
    #     # plt.savefig('sample_image__OrRd.png')
    #     plt.close()


    # for i_batchsize, batch_size in enumerate(batchsizes):
    #     for i_blocksize, block_size in enumerate(blocksizes):
    # for i_start in range(input_image.shape[1]//input_size):
    #     start_input = input_size*i_start
    #     input_xbar = input_image[i_xbar*xbar_size:(i_xbar+1)*xbar_size,start_input:start_input+input_size]
    #     index = range(input_xbar.shape[0]//block_size)
    #     columns = range(input_size//batch_size)
    #     input_blocks_acc = np.zeros((xbar_size//block_size, input_size//batch_size), dtype=int)
    #     # divide the input into 4 blocks, that block_size = 32, and accumulate the block input
    #     #for j_inPeriod in range(input_xbar.shape[1]):
    #     for j_inPeriod in range(input_size):
    #         for i_row in range(xbar_size):
    #             i_block = i_row//block_size
    #             j_batch = j_inPeriod//batch_size
    #             input_blocks_acc[i_block, j_batch] += input_xbar[i_row, j_inPeriod]

    #     print(input_blocks_acc.shape)
    #     # df = DataFrame.from_records()
    #     df = pd.DataFrame(data = input_blocks_acc, columns = columns, index = index)
    #     if batch_size == 8:
    #         center = block_size
    #     elif batch_size == 1:
    #         center = block_size/4
    #     hm, ax = plt.subplots(figsize = (input_size//batch_size,xbar_size//block_size))
    #     hm = sns.heatmap(data = df, square=False, center=center, linewidths=0.1, annot=True, fmt = "d", cmap="RdBu_r") #, center=xbar_size//block_size/4
    #     plt.savefig('image_'+layer+'/batch_{}bit/blocksize_{}/cmap_RdBu_r/input_image{}_xbar{}_start{}.png'.format(batch_size, block_size, i_image, i_xbar, start_input), format = 'png')
    #     hm.clear()
    #     plt.close()

    #     hm, ax = plt.subplots(figsize = (input_size//batch_size,xbar_size//block_size))
    #     hm = sns.heatmap(data = df, square=False, center=center, linewidths=0.1, annot=True, fmt = "d", cmap="OrRd") #, center=xbar_size//block_size/4
    #     plt.savefig('image_'+layer+'/batch_{}bit/blocksize_{}/cmap_OrRd/input_image{}_xbar{}_start{}.png'.format(batch_size, block_size, i_image, i_xbar, start_input), format = 'png')

    #     # hm.close()
    #     plt.close()