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
activatedCounts = []

activatedCounts_sum = np.zeros((4)).astype(int)
categoryCounts_sum = np.zeros((4)).astype(int)
# for now one time execution
for i_image in range(100):
    activeAccumulate_xbar = []
    for i_xbar in range(9): #[0]: 
        
        for i, layer in enumerate(layers):
                
            input_L1_100IMAGE = np.load('input_0/inputConv0_batch_size=20_FP_int32.npy')
            # i_image = 13
            batchsizes = [8] # [1,8]
            blocksizes = [32] # [4, 8 ,16, 32]
            xbar_size = 128
            input_image = input_L1_100IMAGE[i_image]
            # print("input_image.shape")
            # print(input_image.shape)
            # sns.set() 
            # i_xbar = 0
            plot_size = 128
            input_size = plot_size*8*7
            start_input = 0
            block_size = 32

            batch_size = 8
            inputVectors = []
            inputMax0 = []; inputMax1 = []; inputMax2 = []; inputMax3 = []
            inputCategory = [inputMax0, inputMax1, inputMax2, inputMax3]
            # inputCategory = np.asarray(inputCategory)
            inputRearrange = []
            activeAccumulate_local = np.zeros((4)).astype(int)
            for i_plot in range(input_size//plot_size):
                
                input_xbar = input_image[i_xbar*xbar_size:(i_xbar+1)*xbar_size,start_input+plot_size*i_plot:start_input+plot_size*(i_plot+1)]
                
                for i_batchsize, batch_size in enumerate(batchsizes):
                    input_blocks_acc = np.zeros((xbar_size//block_size, plot_size//batch_size), dtype=int)
                    for j_inPeriod in range(plot_size):
                        for i_row in range(xbar_size):
                            i_block = i_row//block_size
                            j_batch = j_inPeriod//batch_size
                            input_blocks_acc[i_block, j_batch] += input_xbar[i_row, j_inPeriod]
                            activeAccumulate_local[i_block] += input_xbar[i_row, j_inPeriod]
                    # for i_blocks_acc in range(input_blocks_acc.shape[1]): # (i_plot*input_blocks_acc.shape[1], (i_plot+1)*input_blocks_acc.shape[1]):
                    #     inputVectors.append(InputVec(i_blocks_acc, threshold, np.array(input_blocks_acc[:,i_blocks_acc])))
                    # print('inputVectors.size:{}'.format(len(inputVectors)))
            
                # print(inputVec.activated_blocks, inputVec.acc_blocks, inputVec.maxVector, 'var = {}, std = {}'.format(inputVec.var, inputVec.std))

            # print(np.array(inputCategory, dtype = object).shape)

        activeAccumulate_xbar.append(activeAccumulate_local)
    # print("image{}, activeAccumulate_xbar with i_xbar = {}\n".format(i_image, i_xbar), activeAccumulate_xbar)
    print("image{}, activeAccumulate_xbar\n".format(i_image), activeAccumulate_xbar)
# print(categoryCounts_sum, activatedCounts_sum)
# np.savetxt('./categoryCounts.csv', categoryCounts, delimiter=',')
# np.savetxt('./activatedCounts.csv', activatedCounts, delimiter=',')