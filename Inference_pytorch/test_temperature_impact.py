# Load the Pandas libraries with alias 'pd' 
import pandas as pd 

# Read data from file 'filename.csv' 
# (in the same directory that your python process is based)
# Control delimiters, rows, column names with read_csv (see later) 
data = pd.read_csv("CORE_DIE_woDRAM_80ns_mlt128x512_avg_pe8_withblockexchange_originalcode_input0.csv", low_memory=False,encoding="utf-8-sig") 
image = data[data['i_image'] == 189]
t_vary = image['T_vary']
# Preview the first 5 lines of the loaded data 
print(t_vary.shape)