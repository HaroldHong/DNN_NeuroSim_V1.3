#from modules.quantize import quantize, quantize_grad, QConv2d, QLinear, RangeBN
import os
import torch.nn as nn
import shutil
from modules.quantization_cpu_np_infer import QConv2d,QLinear
from modules.floatrange_cpu_np_infer import FConv2d, FLinear
import numpy as np
import torch
from utee import wage_quantizer
from utee import float_quantizer

def Neural_Sim(self, input, output): 
    global model_n, FP, i_folder, batch_size
    # print('input\'s len = {}'.format(len(input)) )
    # print('input[0]\'s len = {}'.format(len(input[0])))
    # print(input.shape)
    print("\n quantize layer ", self.name)
    # input_file_name =  './layer_record_' + str(model_n) + '/input' + str(self.name) + '.csv'
    input_file_name =  './layer_record_' + str(model_n) + '/input_{}'.format(i_folder) + '/input' + str(self.name) + 'batch_size={}_FP_int32.npy'.format(batch_size)
    weight_file_name =  './layer_record_' + str(model_n) + '/weight' + str(self.name) + '_WAGE.csv'
    f = open('./layer_record_' + str(model_n) + '/trace_command_batchsize={}.sh'.format(batch_size), "a")
    f.write(weight_file_name+' '+input_file_name+' ')
    if FP:
        weight_q = float_quantizer.float_range_quantize(self.weight,self.wl_weight)
    else:
        weight_q = wage_quantizer.Q(self.weight,self.wl_weight)
    write_matrix_weight( weight_q.cpu().data.numpy(),weight_file_name)
    # print('hook.py: input.len: ', len(input))
    if len(self.weight.shape) > 2:
        k=self.weight.shape[-1]
        write_matrix_activation_conv(stretch_input(input[0].cpu().data.numpy(),k),None,self.wl_input,input_file_name)
    else:
        write_matrix_activation_fc(input[0].cpu().data.numpy(),None ,self.wl_input, input_file_name)

def write_matrix_weight(input_matrix,filename):
    cout = input_matrix.shape[0]
    weight_matrix = input_matrix.reshape(cout,-1).transpose()
    np.savetxt(filename, weight_matrix, delimiter=",",fmt='%10.5f')


def write_matrix_activation_conv(input_matrix,fill_dimension,length,filename):
    filled_matrix_b = np.zeros([input_matrix.shape[0], input_matrix.shape[2],input_matrix.shape[1]*length],dtype=np.str) #(27,900*8)
    for i in range(input_matrix.shape[0]): # 500 individual images
        filled_matrix_bin,scale = dec2bin(input_matrix[i,:],length) #only take the first data sample
        # print('input_matrix\'s shape')
        # print(input_matrix.shape)
        # print('filled_matrix_b\'s shape')
        # print(filled_matrix_b.shape)
        # print('filled_matrix_bin[0]\'s len')
        # print(len(filled_matrix_bin[0]))

        # print('scale')
        # print(scale)
        # for i in input_matrix.shape[0]: # 500 individual images
        filled_matrix_cpy = filled_matrix_bin.copy()
        for j,b in enumerate(filled_matrix_cpy):
            filled_matrix_b[i,:,j::length] = b.transpose()
    filled_matrix_int32 = filled_matrix_b.astype(np.int32)
    activity = np.sum(filled_matrix_int32, axis=None)/np.size(filled_matrix_int32)
    # arr_int = arr_u1.astype(np.int32)
    np.save(filename, filled_matrix_int32)
    # np.savetxt(filename, filled_matrix_b, delimiter=",",fmt='%s')
    return activity

# def write_matrix_activation_conv_bak(input_matrix,fill_dimension,length,filename):
#     filled_matrix_b = np.zeros([input_matrix.shape[2],input_matrix.shape[1]*length],dtype=np.str) #(27,900*8)
#     filled_matrix_bin,scale = dec2bin(input_matrix[0,:],length) #only take the first data sample
#     print('input_matrix\'s shape')
#     print(input_matrix.shape)
#     print('filled_matrix_b\'s shape')
#     print(filled_matrix_b.shape)
#     print('filled_matrix_bin[0]\'s len')
#     print(len(filled_matrix_bin[0]))
#     # print('scale')
#     # print(scale)
#     for i,b in enumerate(filled_matrix_bin):
#         filled_matrix_b[:,i::length] =  b.transpose()
#     np.savetxt(filename, filled_matrix_b, delimiter=",",fmt='%s')

def write_matrix_activation_fc(input_matrix,fill_dimension,length,filename):
    print('write_matrix_activation_fc')
    filled_matrix_b = np.zeros([input_matrix.shape[0], input_matrix.shape[1],length],dtype=np.str)
    for i in range(input_matrix.shape[0]): # 500 individual images
        filled_matrix_bin,scale = dec2bin(input_matrix[i,:],length)
        # filled_matrix_cpy = filled_matrix_bin.copy()

        for j,b in enumerate(filled_matrix_bin):
            filled_matrix_b[i,:,j] =  b
    filled_matrix_int32 = filled_matrix_b.astype(np.int32)
    activity = np.sum(filled_matrix_int32, axis=None)/np.size(filled_matrix_int32)
    # np.savetxt(filename, filled_matrix_b, delimiter=",",fmt='%s')
    np.save(filename, filled_matrix_int32)
    return activity

def stretch_input(input_matrix,window_size = 5): # no padding?
    

    # without padding
    # input_shape = input_matrix.shape
    # item_num = (input_shape[2] - window_size + 1) * (input_shape[3]-window_size + 1)
    # output_matrix = np.zeros((input_shape[0],item_num,input_shape[1]*window_size*window_size))
    # iter = 0
    # for i in range( input_shape[2]-window_size + 1 ):
    #     for j in range( input_shape[3]-window_size + 1 ):
    #         for b in range(input_shape[0]):
    #             output_matrix[b,iter,:] = input_matrix[b, :, i:i+window_size,j: j+window_size].reshape(input_shape[1]*window_size*window_size)
    #         iter += 1
    # # print('output_matrix\'s shape')
    # print("\ninput_matrix.shape: ", input_matrix.shape)
    # print("\nafter stretch: ", output_matrix.shape)
    # return output_matrix

    # add padding to input stretching

    input_shape = input_matrix.shape
    # print('\ninput_matrix\'s shape')
    
    input_matrix_padded = np.zeros((input_shape[0], input_shape[1], input_shape[2]+2, input_shape[3]+2)) # default padding is 1
    input_matrix_padded[:,:,1:input_shape[2]+1,1:input_shape[3]+1] = input_matrix
    input_shape_padded = input_matrix_padded.shape
    item_num = (input_shape_padded[2] - window_size + 1) * (input_shape_padded[3]-window_size + 1)
    # eg. pad the 32x32 feature map to 34x34, with original matrix in central
    output_matrix = np.zeros((input_shape_padded[0],item_num,input_shape_padded[1]*window_size*window_size))
    iter = 0
    for i in range( input_shape_padded[2]-window_size + 1 ):
        for j in range( input_shape_padded[3]-window_size + 1 ):
            for b in range(input_shape_padded[0]):
                output_matrix[b,iter,:] = input_matrix_padded[b, :, i:i+window_size,j: j+window_size].reshape(input_shape_padded[1]*window_size*window_size)
            iter += 1
    # print('output_matrix\'s shape')
    print("\ninput_matrix.shape: ", input_matrix.shape)
    print("after stretch: ", output_matrix.shape)
    print('\n')
    return output_matrix

def dec2bin(x,n):
    y = x.copy()
    out = []
    scale_list = []
    delta = 1.0/(2**(n-1))
    x_int = x/delta
    # print('x\'s shape:')
    # print(x.shape)
    # print('x_int:')
    # print(x_int)
    base = 2**(n-1)
    # sign bit
    y[x_int>=0] = 0
    y[x_int< 0] = 1
    # print('y:')
    # print(y)
    rest = x_int + base*y
    # print('rest:')
    # print(rest)
    out.append(y.copy())
    scale_list.append(-base*delta)
    for i in range(n-1):
        base = base/2
        y[rest>=base] = 1
        y[rest<base]  = 0
        
        rest = rest - base * y
        # print('i = {}, rest:'.format(i+1))
        # print(rest)
        out.append(y.copy())
        scale_list.append(base * delta)
    # print('out\'s len:')
    # print(len(out))
    return out,scale_list

def bin2dec(x,n):
    bit = x.pop(0)
    base = 2**(n-1)
    delta = 1.0/(2**(n-1))
    y = -bit*base
    base = base/2
    for bit in x:
        y = y+base*bit
        base= base/2
    out = y*delta
    return out

def remove_hook_list(hook_handle_list):
    for handle in hook_handle_list:
        handle.remove()

def hardware_evaluation(model,wl_weight,wl_activation,model_name,mode,folder_i,args): 
    global model_n, i_folder, batch_size, FP
    model_n = model_name
    i_folder = folder_i
    batch_size = args.batch_size
    
    FP = 1 if mode=='FP' else 0
    print('{}th input data'.format(folder_i))
    hook_handle_list = []

    # if not os.path.exists('./layer_record_'+str(model_name)+'/input_{}'.format(folder_i)):
    #     os.makedirs('./layer_record_'+str(model_name)+'/input_{}'.format(folder_i))
    # if os.path.exists('./layer_record_'+str(model_name)+'/input_{}/trace_command_batchsize={}.sh'.format(folder_i, batch_size)):
    #     os.remove('./layer_record_'+str(model_name)+'/input_{}/trace_command_batchsize={}.sh'.format(folder_i, batch_size))
    # f = open('./layer_record_'+str(model_name)+'/input_{}/trace_command_batchsize={}.sh'.format(folder_i, batch_size), "w")
    # f.write('./NeuroSIM/main ./NeuroSIM/NetWork_'+str(model_name)+'.csv '+str(wl_weight)+' '+str(wl_activation)+' ')
    for i, layer in enumerate(model.modules()):
        if isinstance(layer, (FConv2d, QConv2d, nn.Conv2d)) or isinstance(layer, (FLinear, QLinear, nn.Linear)):
            hook_handle_list.append(layer.register_forward_hook(Neural_Sim))
    return hook_handle_list