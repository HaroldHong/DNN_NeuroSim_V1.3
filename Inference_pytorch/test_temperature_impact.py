# Load the Pandas libraries with alias 'pd' 
import pandas as pd 
import numpy as np
# Read data from file 'filename.csv' 
# (in the same directory that your python process is based)
# Control delimiters, rows, column names with read_csv (see later) 
data = pd.read_csv("CORE_DIE_woDRAM_80ns_mlt128x512_avg_pe8_woblockexchange_originalcode_input0.csv", low_memory=False,encoding="utf-8-sig") 

range_avg = np.zeros(max(data['i_image'])+1)
range_avg_my = np.zeros(max(data['i_image'])+1)
for i_image in range(range_avg.shape[0]):
    image = data[data['i_image'] == i_image]
    range_avg[i_image] += image['range_Temperature'].mean(axis=0)
    image_rangeT = image['range_Temperature']
    # print(image_rangeT)
    # for i_input in range(image_rangeT.shape[0]):
    #     range_avg_my[i_image] += image_rangeT.loc[i_input]
    # range_avg_my[i_image] /= image.shape[0]
    
maxIndexes = np.argsort(range_avg)[::-1]
range_sort = np.sort(range_avg)[::-1]
print(maxIndexes[:30])
print(range_sort[:30])