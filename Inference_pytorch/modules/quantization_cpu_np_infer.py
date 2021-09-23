import torch
import torch.nn as nn
import torch.nn.functional as F
from utee import wage_initializer,wage_quantizer, I_V_T_smallSim
# from utee.w2g import w2g
from torch._jit_internal import weak_script_method
import numpy as np
import pandas as pd
import math
from multiprocessing import Pool

class QConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,logger = None,clip_weight = False,wage_init=False,quantize_weight= False,clip_output =False,quantize_output = False,
                 wl_input =8,wl_activate=8,wl_error=8,wl_weight= 8,inference=0,onoffratio=10,cellBit=1,subArray=128,ADCprecision=5,vari=0,t=0,v=0,detect=0,target=0,debug = 0, name = 'Qconv', model = None, layer_Conv = 0):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)
        self.logger = logger
        self.clip_weight = clip_weight
        self.wage_init = wage_init
        self.quantize_weight = quantize_weight
        self.clip_output = clip_output
        self.debug = debug
        self.wl_weight = wl_weight
        self.quantize_output = quantize_output
        self.wl_activate = wl_activate
        self.wl_error = wl_error
        self.wl_input = wl_input
        self.inference = inference
        self.onoffratio = onoffratio
        self.cellBit = cellBit
        self.subArray = subArray
        self.ADCprecision = ADCprecision
        self.vari = vari
        self.t = t
        self.v = v
        self.detect = detect
        self.target = target
        self.name = name
        self.model = model
        self.layer_Conv = layer_Conv
        # pytorx parameters
        weight_flatten = self.weight.view(self.out_channels, -1)
        self.crxb_row, self.crxb_row_pads = self.num_pad(
            weight_flatten.shape[1], self.subArray)
        self.crxb_col, self.crxb_col_pads = self.num_pad(
            weight_flatten.shape[0], self.subArray)
        self.h_out = None
        self.w_out = None
        self.w_pad = (0, self.crxb_row_pads, 0, self.crxb_col_pads)
        self.input_pad = (0, 0, 0, self.crxb_row_pads)
        weight_padded = F.pad(weight_flatten, self.w_pad,
                              mode='constant', value=0)
        weight_crxb = weight_padded.view(self.crxb_col, self.subArray,
                                         self.crxb_row, self.subArray).transpose(1, 2)
        self.Gmax = 1 #/(13e3)  # max conductance
        self.Gmin = self.Gmax/onoffratio  # min conductance
        self.delta_g = (self.Gmax - self.Gmin) / (2 ** 7)  # conductance step
        # self.w2g = w2g(self.delta_g, Gmin=self.Gmin, G_SA0=self.Gmax,
        #                G_SA1=self.Gmin, weight_shape=weight_crxb.shape, enable_SAF=False)        
        self.scale  = wage_initializer.wage_init_(self.weight, self.wl_weight, factor=1.0)
        # self.nchout_index = nn.Parameter(torch.arange(self.out_channels), requires_grad=False)
        self.nchout_index = torch.arange(self.out_channels).cuda()

    def num_pad(self, source, target):
        crxb_index = math.ceil(source / target)
        num_padding = crxb_index * target - source
        return crxb_index, num_padding

    @weak_script_method
    def forward(self, input):
        
        weight1 = self.weight * self.scale + (self.weight - self.weight * self.scale).detach()
        weight = weight1 + (wage_quantizer.Q(weight1,self.wl_weight) -weight1).detach()
        outputOrignal= F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        print("\nself.weight.shape: ", self.weight.shape)
        print("weight1.shape: ", weight1.shape)
        print("weight.shape: ", weight.shape, ", self.subArray = ", self.subArray)
        print("input.shape: ", input.shape, ", outputOrignal.shape = ", outputOrignal.shape)
        bitWeight = int(self.wl_weight)
        bitActivation = int(self.wl_input)

        if self.inference == 1 and self.model=='VGG8':
            # set parameters for Hardware Inference
            onoffratio = self.onoffratio
            upper = 1
            lower = 1/onoffratio
        
            output = torch.zeros_like(outputOrignal)
            del outputOrignal
            cellRange = 2**self.cellBit   # cell precision is 1 for SLC
        
            # Now consider on/off ratio
            dummyP = torch.zeros_like(weight)
            dummyP[:,:,:,:] = (cellRange-1)*(upper+lower)/2
            
            # args in pytorX
            weight_flatten = self.weight.view(self.out_channels, -1)
            self.crxb_row, self.crxb_row_pads = self.num_pad(
                weight_flatten.shape[1], self.subArray)
            self.crxb_col, self.crxb_col_pads = self.num_pad(
                weight_flatten.shape[0], self.subArray)
            w_pad = (0, self.crxb_row_pads, 0, self.crxb_col_pads)
            input_pad = (0, 0, 0, self.crxb_row_pads)
            
            if self.h_out is None and self.w_out is None:
                self.h_out = int(
                    (input.shape[2] - self.kernel_size[0] + 2 * self.padding[0]) / self.stride[0] + 1)
                self.w_out = int(
                    (input.shape[3] - self.kernel_size[0] + 2 * self.padding[0]) / self.stride[0] + 1)

            print("\npyTorX layer_Conv", self.layer_Conv, ": \nweight_flatten.shape:", weight_flatten.shape)
            print("crxb_row:", self.crxb_row, ", crxb_row_pads:", self.crxb_row_pads)
            print("crxb_col:", self.crxb_col, ", crxb_col_pads:", self.crxb_col_pads)
            print("w_pad:", w_pad, ", input_pad:", input_pad, "int(weight.shape[1]/self.subArray) = ", int(weight.shape[1]/self.subArray))

            for i in range (self.weight.shape[2]):
                for j in range (self.weight.shape[3]):
                    # need to divide to different subArray
                    numSubArray = int(weight.shape[1]/self.subArray)
                    # cut into different subArrays
                    if numSubArray == 0:
                        mask = torch.zeros_like(weight)
                        mask[:,:,i,j] = 1
                        if weight.shape[1] == 3:
                            # after get the spacial kernel, need to transfer floating weight [-1, 1] to binarized ones
                            X_decimal = torch.round((2**bitWeight - 1)/2 * (weight+1) + 0)*mask
                            outputP = torch.zeros_like(output)
                            outputD = torch.zeros_like(output)
                            for k in range (int(bitWeight/self.cellBit)):
                                remainder = torch.fmod(X_decimal, cellRange)*mask
                                # retention
                                remainder = wage_quantizer.Retention(remainder,self.t,self.v,self.detect,self.target)
                                variation = np.random.normal(0, self.vari, list(weight.size())).astype(np.float32)
                                X_decimal = torch.round((X_decimal-remainder)/cellRange)*mask
                                # Now also consider weight has on/off ratio effects
                                # Here remainder is the weight mapped to Hardware, so we introduce on/off ratio in this value
                                # the range of remainder is [0, cellRange-1], we truncate it to [lower, upper]
                                remainderQ = (upper-lower)*(remainder-0)+(cellRange-1)*lower   # weight cannot map to 0, but to Gmin
                                remainderQ = remainderQ + remainderQ*torch.from_numpy(variation).cuda()
                                outputPartial= F.conv2d(input, remainderQ*mask, self.bias, self.stride, self.padding, self.dilation, self.groups)
                                outputDummyPartial= F.conv2d(input, dummyP*mask, self.bias, self.stride, self.padding, self.dilation, self.groups)
                                scaler = cellRange**k
                                outputP = outputP + outputPartial*scaler*2/(1-1/onoffratio)
                                outputD = outputD + outputDummyPartial*scaler*2/(1-1/onoffratio)
                            outputP = outputP - outputD
                            output = output + outputP
                        else:
                            # print("input.shape: ", input.shape)
                            # quantize input into binary sequence
                            inputQ = torch.round((2**bitActivation - 1)/1 * (input-0) + 0)
                            outputIN = torch.zeros_like(output)
                            # print("inputB.shape for F.Conv2d function: ", input.shape)
                            # print("output.shape from F.Conv2d function: ", output.shape)
                            for z in range(bitActivation):
                                inputB = torch.fmod(inputQ, 2)
                                inputQ = torch.round((inputQ-inputB)/2)
                                outputP = torch.zeros_like(output)
                                
                                # after get the spacial kernel, need to transfer floating weight [-1, 1] to binarized ones
                                X_decimal = torch.round((2**bitWeight - 1)/2 * (weight+1) + 0)*mask
                                outputD = torch.zeros_like(output)
                                for k in range (int(bitWeight/self.cellBit)):
                                    remainder = torch.fmod(X_decimal, cellRange)*mask
                                    # retention
                                    remainder = wage_quantizer.Retention(remainder,self.t,self.v,self.detect,self.target)
                                    variation = np.random.normal(0, self.vari, list(weight.size())).astype(np.float32)
                                    X_decimal = torch.round((X_decimal-remainder)/cellRange)*mask
                                    # Now also consider weight has on/off ratio effects
                                    # Here remainder is the weight mapped to Hardware, so we introduce on/off ratio in this value
                                    # the range of remainder is [0, cellRange-1], we truncate it to [lower, upper]
                                    remainderQ = (upper-lower)*(remainder-0)+(cellRange-1)*lower   # weight cannot map to 0, but to Gmin
                                    remainderQ = remainderQ + remainderQ*torch.from_numpy(variation).cuda()
                                    
                                    
                                    outputPartial= F.conv2d(inputB, remainderQ*mask, self.bias, self.stride, self.padding, self.dilation, self.groups)
                                    outputDummyPartial= F.conv2d(inputB, dummyP*mask, self.bias, self.stride, self.padding, self.dilation, self.groups)
                                    # Add ADC quanization effects here !!!
                                    outputPartialQ = wage_quantizer.LinearQuantizeOut(outputPartial, self.ADCprecision)
                                    outputDummyPartialQ = wage_quantizer.LinearQuantizeOut(outputDummyPartial, self.ADCprecision)
                                    scaler = cellRange**k
                                    outputP = outputP + outputPartialQ*scaler*2/(1-1/onoffratio)
                                    outputD = outputD + outputDummyPartialQ*scaler*2/(1-1/onoffratio)
                                scalerIN = 2**z
                                outputIN = outputIN + (outputP - outputD)*scalerIN
                            output = output + outputIN/(2**bitActivation)
                    else:
                        # # quantize input into binary sequence
                        # inputQ = torch.round((2**bitActivation - 1)/1 * (input-0) + 0)
                        # outputIN = torch.zeros_like(output)
                        # for z in range(bitActivation):
                        #     inputB = torch.fmod(inputQ, 2)
                        #     inputQ = torch.round((inputQ-inputB)/2)
                        #     outputP = torch.zeros_like(output)
                        #     for s in range(numSubArray):
                        #         mask = torch.zeros_like(weight)
                        #         mask[:,(s*self.subArray):(s+1)*self.subArray, i, j] = 1
                        #         # after get the spacial kernel, need to transfer floating weight [-1, 1] to binarized ones
                        #         X_decimal = torch.round((2**bitWeight - 1)/2 * (weight+1) + 0)*mask
                        #         outputSP = torch.zeros_like(output)
                        #         outputD = torch.zeros_like(output)
                        #         for k in range (int(bitWeight/self.cellBit)):
                        #             remainder = torch.fmod(X_decimal, cellRange)*mask
                        #             # retention
                        #             remainder = wage_quantizer.Retention(remainder,self.t,self.v,self.detect,self.target)
                        #             variation = np.random.normal(0, self.vari, list(weight.size())).astype(np.float32)
                        #             X_decimal = torch.round((X_decimal-remainder)/cellRange)*mask
                        #             # Now also consider weight has on/off ratio effects
                        #             # Here remainder is the weight mapped to Hardware, so we introduce on/off ratio in this value
                        #             # the range of remainder is [0, cellRange-1], we truncate it to [lower, upper]*(cellRange-1)
                        #             remainderQ = (upper-lower)*(remainder-0)+(cellRange-1)*lower   # weight cannot map to 0, but to Gmin
                        #             remainderQ = remainderQ + remainderQ*torch.from_numpy(variation).cuda()
                                    
                        #             # modified alike pytorX
                                    
                        #             input_unfold = F.unfold(inputB, kernel_size=self.kernel_size,
                        #             dilation=self.dilation, padding=self.padding,
                        #             stride=self.stride)
                        #             weight_flatten = (remainderQ*mask).view(self.out_channels, -1)
                                    
                        #             # 2.2. add paddings
                        #             weight_padded = F.pad(weight_flatten, w_pad,
                        #                                 mode='constant', value=0)
                        #             input_padded = F.pad(input_unfold, input_pad,
                        #                                 mode='constant', value=0)
                        #             # 2.3. reshape to crxb size
                        #             input_crxb = input_padded.view(input.shape[0], 1, self.crxb_row,
                        #                                         self.subArray, input_padded.shape[2])
                        #             weight_crxb = weight_padded.view(self.crxb_col, self.subArray,
                        #                                             self.crxb_row, self.subArray).transpose(1, 2)
                        #             # G_crxb = self.w2g(weight_crxb)

                        #             # if z==s==k==0:
                        #             #     print("\n inputB.shape: ", inputB.shape, "flatten and unfold \ninput_unfold.shape: ", input_unfold.shape, " weight_flatten.shape: ", weight_flatten.shape, 
                        #             #     " input_crxb.shape: ", input_crxb.shape, " weight_crxb.shape: ", weight_crxb.shape) #, " G_crxb.shape: ", G_crxb.shape)

                        #             output_crxb_standard = torch.matmul(weight_crxb, input_crxb)
                        #             output_crxb = torch.zeros_like(output_crxb_standard)
                        #             # decompose matrix multiplication
                        #             for in_0 in range(input_crxb.shape[0]): # in_0 is batch size of images
                        #                 for w_0 in range(weight_crxb.shape[0]): # w_0 is the first dimention of weight_crxb
                        #                     for in_4 in range(input_crxb.shape[4]): #i_4 is the last dimention
                        #                         # sub_input_unsqueeze = (input_crxb[i_0,i_1,:,:,i_4]).unsqueeze(2).unsqueeze(0).unsqueeze(0)
                        #                         output_crxb[in_0,w_0,:,:,in_4] += torch.matmul(weight_crxb[w_0,:,:,:], input_crxb[in_0,0,:,:,in_4].unsqueeze(2)).squeeze(2)# torch.matmul(weight_crxb, sub_input_unsqueeze)
                        #             deviation_max = torch.max(output_crxb - output_crxb_standard).item()
                        #             deviation_min = torch.min(output_crxb - output_crxb_standard).item()
                        #             output_crxb_standard_max = torch.max(output_crxb_standard).item()
                        #             output_crxb_standard_min = torch.min(output_crxb_standard).item()
                        #             if abs(deviation_max) > 1:
                        #                 print("deviation of mul decompose: ", deviation_max, "min: ", deviation_min, " max of standard: ", output_crxb_standard_max, " min: ", output_crxb_standard_min)


                        #             output_sum = torch.sum(output_crxb, dim=2)
                        #             outputPartial = output_sum.view(output_sum.shape[0],
                        #                                     output_sum.shape[1] * output_sum.shape[2],
                        #                                     self.h_out,
                        #                                     self.w_out).index_select(dim=1, index=self.nchout_index)
                        #             # outputPartial *= 2.8
                        #             # end of pytorx convolutional computation

                        #             # outputPartial= F.conv2d(inputB, remainderQ*mask, self.bias, self.stride, self.padding, self.dilation, self.groups)
                        #             outputDummyPartial= F.conv2d(inputB, dummyP*mask, self.bias, self.stride, self.padding, self.dilation, self.groups)
                        #             # Add ADC quanization effects here !!!
                        #             outputPartialQ = wage_quantizer.LinearQuantizeOut(outputPartial, self.ADCprecision)
                        #             outputDummyPartialQ = wage_quantizer.LinearQuantizeOut(outputDummyPartial, self.ADCprecision)
                        #             scaler = cellRange**k
                        #             outputSP = outputSP + outputPartialQ*scaler*2/(1-1/onoffratio)
                        #             outputD = outputD + outputDummyPartialQ*scaler*2/(1-1/onoffratio)
                        #         # !!! Important !!! the dummy need to be multiplied by a ratio
                        #         outputSP = outputSP - outputD  # minus dummy column
                        #         outputP = outputP + outputSP
                        #     scalerIN = 2**z
                        #     outputIN = outputIN + outputP*scalerIN
                        
                        # quantize input into binary sequence
                        inputQ = torch.round((2**bitActivation - 1)/1 * (input-0) + 0)
                        outputIN = torch.zeros_like(output)
                        for z in range(bitActivation):
                            inputB = torch.fmod(inputQ, 2)
                            inputQ = torch.round((inputQ-inputB)/2)
                            outputP = torch.zeros_like(output)
                            for s in range(numSubArray):
                                mask = torch.zeros_like(weight)
                                mask[:,(s*self.subArray):(s+1)*self.subArray, i, j] = 1
                                # after get the spacial kernel, need to transfer floating weight [-1, 1] to binarized ones
                                X_decimal = torch.round((2**bitWeight - 1)/2 * (weight+1) + 0)*mask
                                outputSP = torch.zeros_like(output)
                                outputD = torch.zeros_like(output)
                                for k in range (int(bitWeight/self.cellBit)):
                                    remainder = torch.fmod(X_decimal, cellRange)*mask
                                    # retention
                                    remainder = wage_quantizer.Retention(remainder,self.t,self.v,self.detect,self.target)
                                    variation = np.random.normal(0, self.vari, list(weight.size())).astype(np.float32)
                                    X_decimal = torch.round((X_decimal-remainder)/cellRange)*mask
                                    # Now also consider weight has on/off ratio effects
                                    # Here remainder is the weight mapped to Hardware, so we introduce on/off ratio in this value
                                    # the range of remainder is [0, cellRange-1], we truncate it to [lower, upper]*(cellRange-1)
                                    remainderQ = (upper-lower)*(remainder-0)+(cellRange-1)*lower   # weight cannot map to 0, but to Gmin
                                    remainderQ = remainderQ + remainderQ*torch.from_numpy(variation).cuda()
                                    
                                    # modified alike pytorX
                                    
                                    input_unfold = F.unfold(inputB, kernel_size=self.kernel_size,
                                    dilation=self.dilation, padding=self.padding,
                                    stride=self.stride)
                                    weight_flatten = (remainderQ*mask).view(self.out_channels, -1)
                                    
                                    # 2.2. add paddings
                                    weight_padded = F.pad(weight_flatten, w_pad,
                                                        mode='constant', value=0)
                                    input_padded = F.pad(input_unfold, input_pad,
                                                        mode='constant', value=0)
                                    # 2.3. reshape to crxb size
                                    input_crxb = input_padded.view(input.shape[0], 1, self.crxb_row,
                                                                self.subArray, input_padded.shape[2])
                                    weight_crxb = weight_padded.view(self.crxb_col, self.subArray,
                                                                    self.crxb_row, self.subArray).transpose(1, 2)
                                    # G_crxb = self.w2g(weight_crxb)

                                    # if z==s==k==0:
                                    #     print("\n inputB.shape: ", inputB.shape, "flatten and unfold \ninput_unfold.shape: ", input_unfold.shape, " weight_flatten.shape: ", weight_flatten.shape, 
                                    #     " input_crxb.shape: ", input_crxb.shape, " weight_crxb.shape: ", weight_crxb.shape) #, " G_crxb.shape: ", G_crxb.shape)

                                    # Start decomposation
                                    # output_crxb_standard = torch.matmul(weight_crxb, input_crxb)
                                    # output_crxb = torch.zeros_like(output_crxb_standard)
                                    # # decompose matrix multiplication
                                    # for in_0 in range(input_crxb.shape[0]): # in_0 is batch size of images
                                    #     for w_0 in range(weight_crxb.shape[0]): # w_0 is the first dimention of weight_crxb
                                    #         for in_4 in range(input_crxb.shape[4]): #i_4 is the last dimention
                                    #             # sub_input_unsqueeze = (input_crxb[i_0,i_1,:,:,i_4]).unsqueeze(2).unsqueeze(0).unsqueeze(0)
                                    #             output_crxb[in_0,w_0,:,:,in_4] += torch.matmul(weight_crxb[w_0,:,:,:], input_crxb[in_0,0,:,:,in_4].unsqueeze(2)).squeeze(2)# torch.matmul(weight_crxb, sub_input_unsqueeze)
                                    # deviation_max = torch.max(output_crxb - output_crxb_standard).item()
                                    # deviation_min = torch.min(output_crxb - output_crxb_standard).item()
                                    # output_crxb_standard_max = torch.max(output_crxb_standard).item()
                                    # output_crxb_standard_min = torch.min(output_crxb_standard).item()
                                    # if abs(deviation_max) > 1:
                                    #     print("deviation of mul decompose: ", deviation_max, "min: ", deviation_min, " max of standard: ", output_crxb_standard_max, " min: ", output_crxb_standard_min)

                                    # output_sum = torch.sum(output_crxb, dim=2)
                                    # outputPartial = output_sum.view(output_sum.shape[0],
                                    #                         output_sum.shape[1] * output_sum.shape[2],
                                    #                         self.h_out,
                                    #                         self.w_out).index_select(dim=1, index=self.nchout_index)
                                    
                                    # outputPartial *= 2.8
                                    # end of pytorx convolutional computation

                                    outputPartial= F.conv2d(inputB, remainderQ*mask, self.bias, self.stride, self.padding, self.dilation, self.groups)
                                    outputDummyPartial= F.conv2d(inputB, dummyP*mask, self.bias, self.stride, self.padding, self.dilation, self.groups)
                                    # Add ADC quanization effects here !!!
                                    outputPartialQ = wage_quantizer.LinearQuantizeOut(outputPartial, self.ADCprecision)
                                    outputDummyPartialQ = wage_quantizer.LinearQuantizeOut(outputDummyPartial, self.ADCprecision)
                                    scaler = cellRange**k
                                    outputSP = outputSP + outputPartialQ*scaler*2/(1-1/onoffratio)
                                    outputD = outputD + outputDummyPartialQ*scaler*2/(1-1/onoffratio)
                                # !!! Important !!! the dummy need to be multiplied by a ratio
                                outputSP = outputSP - outputD  # minus dummy column
                                outputP = outputP + outputSP
                            scalerIN = 2**z
                            outputIN = outputIN + outputP*scalerIN

                        output = output + outputIN/(2**bitActivation)
            output = output/(2**bitWeight)   # since weight range was convert from [-1, 1] to [-256, 256]
            print("input.shape: ", input.shape, ", output.shape: ", output.shape)
            # if self.layer_Conv == 1:    # Conv layer 1 gets impacted most
                # print("output before and after the thermal imbalance impact: output_b = ", output, ", output_a = ", output*1.2)
                # output *= 400 
                # print(output.shape)
        elif self.inference == 1:
            weight1 = self.weight * self.scale + (self.weight - self.weight * self.scale).detach()
            weight = weight1 + (wage_quantizer.Q(weight1,self.wl_weight) -weight1).detach()
            weight = wage_quantizer.Retention(weight,self.t,self.v,self.detect,self.target)
            input = wage_quantizer.Q(input,self.wl_input)
            output= F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            output = wage_quantizer.LinearQuantizeOut(output, self.ADCprecision)
        else:
            # original WAGE QCov2d
            weight1 = self.weight * self.scale + (self.weight - self.weight * self.scale).detach()
            weight = weight1 + (wage_quantizer.Q(weight1,self.wl_weight) -weight1).detach()
            weight = wage_quantizer.Retention(weight,self.t,self.v,self.detect,self.target)
            output= F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        output = output/self.scale
        output = wage_quantizer.WAGEQuantizer_f(output, self.wl_activate, self.wl_error)
        
        return output

# Quantized Convolutional layer with Temperature impacts. Temperature recordings are in temperatures_images
class QConv2d_T(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,logger = None,clip_weight = False,wage_init=False,quantize_weight= False,clip_output =False,quantize_output = False,
                 wl_input =8,wl_activate=8,wl_error=8,wl_weight= 8,inference=0,onoffratio=10,cellBit=1,subArray=128,ADCprecision=5,vari=0,t=0,v=0,detect=0,target=0,debug = 0, name = 'Qconv',
                 model = None, layer_Conv = 0, indexs_high_t_range=None, temperatures_images = None):
        super(QConv2d_T, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)
        self.logger = logger
        self.clip_weight = clip_weight
        self.wage_init = wage_init
        self.quantize_weight = quantize_weight
        self.clip_output = clip_output
        self.debug = debug
        self.wl_weight = wl_weight
        self.quantize_output = quantize_output
        self.wl_activate = wl_activate
        self.wl_error = wl_error
        self.wl_input = wl_input
        self.inference = inference
        self.onoffratio = onoffratio
        self.cellBit = cellBit
        self.subArray = subArray
        self.ADCprecision = ADCprecision
        self.vari = vari
        self.t = t
        self.v = v
        self.detect = detect
        self.target = target
        self.name = name
        self.model = model
        self.layer_Conv = layer_Conv
        self.scale  = wage_initializer.wage_init_(self.weight, self.wl_weight, factor=1.0)

        # pytorx parameters
        weight_flatten = self.weight.view(self.out_channels, -1)
        self.crxb_row, self.crxb_row_pads = self.num_pad(
            weight_flatten.shape[1], self.subArray)
        self.crxb_col, self.crxb_col_pads = self.num_pad(
            weight_flatten.shape[0], self.subArray)
        self.h_out = None
        self.w_out = None
        self.w_pad = (0, self.crxb_row_pads, 0, self.crxb_col_pads)
        self.input_pad = (0, 0, 0, self.crxb_row_pads)
        weight_padded = F.pad(weight_flatten, self.w_pad,
                              mode='constant', value=0)
        weight_crxb = weight_padded.view(self.crxb_col, self.subArray,
                                         self.crxb_row, self.subArray).transpose(1, 2)
        self.Gmax = 1 #/(13e3)  # max conductance
        self.Gmin = self.Gmax/onoffratio  # min conductance
        self.delta_g = (self.Gmax - self.Gmin) / (2 ** 7)  # conductance step
        # self.w2g = w2g(self.delta_g, Gmin=self.Gmin, G_SA0=self.Gmax,
        #                G_SA1=self.Gmin, weight_shape=weight_crxb.shape, enable_SAF=False)        
        
        # self.nchout_index = nn.Parameter(torch.arange(self.out_channels), requires_grad=False)
        self.nchout_index = torch.arange(self.out_channels).cuda()

        # Temperature recordings of selected images, Conv layer 1 and 8th PE by default
        self.temperature_sim = False
        self.indexs_high_t_range = indexs_high_t_range
        self.temperatures_images = temperatures_images
        self.numblock = 4
        self.blocksize = self.subArray/self.numblock
    def num_pad(self, source, target):
        crxb_index = math.ceil(source / target)
        num_padding = crxb_index * target - source
        return crxb_index, num_padding

    @weak_script_method
    def forward(self, input):
        
        weight1 = self.weight * self.scale + (self.weight - self.weight * self.scale).detach()
        weight = weight1 + (wage_quantizer.Q(weight1,self.wl_weight) -weight1).detach()
        outputOrignal= F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        print("\nself.weight.shape: ", self.weight.shape)
        print("weight1.shape: ", weight1.shape)
        print("weight.shape: ", weight.shape, ", self.subArray = ", self.subArray)
        print("input.shape: ", input.shape, ", outputOrignal.shape = ", outputOrignal.shape)
        bitWeight = int(self.wl_weight)
        bitActivation = int(self.wl_input)

        if self.inference == 1 and self.model=='VGG8':
            # set parameters for Hardware Inference
            onoffratio = self.onoffratio
            upper = 1
            lower = upper/onoffratio
        
            output = torch.zeros_like(outputOrignal)
            del outputOrignal
            cellRange = 2**self.cellBit   # cell precision is 1 for SLC
        
            # Now consider on/off ratio
            dummyP = torch.zeros_like(weight)
            dummyP[:,:,:,:] = (cellRange-1)*(1+0.1)/2
            # dummyP[:,:,:,:] = (cellRange-1)*(upper+lower)/2

            # args in pytorX
            weight_flatten = self.weight.view(self.out_channels, -1)
            self.crxb_row, self.crxb_row_pads = self.num_pad(
                weight_flatten.shape[1], self.subArray)
            self.crxb_col, self.crxb_col_pads = self.num_pad(
                weight_flatten.shape[0], self.subArray)
            w_pad = (0, self.crxb_row_pads, 0, self.crxb_col_pads)
            input_pad = (0, 0, 0, self.crxb_row_pads)
            
            if self.h_out is None and self.w_out is None:
                self.h_out = int(
                    (input.shape[2] - self.kernel_size[0] + 2 * self.padding[0]) / self.stride[0] + 1)
                self.w_out = int(
                    (input.shape[3] - self.kernel_size[0] + 2 * self.padding[0]) / self.stride[0] + 1)

            print("\npyTorX layer_Conv", self.layer_Conv, ": \nweight_flatten.shape:", weight_flatten.shape)
            print("crxb_row:", self.crxb_row, ", crxb_row_pads:", self.crxb_row_pads)
            print("crxb_col:", self.crxb_col, ", crxb_col_pads:", self.crxb_col_pads)
            print("w_pad:", w_pad, ", input_pad:", input_pad, "int(weight.shape[1]/self.subArray) = ", int(weight.shape[1]/self.subArray))

            # load in temperature maps of 8th PE
            if self.layer_Conv == 1:
                T_map_1tile = np.zeros((input.shape[0],input.shape[2]*input.shape[3],4))
                for index, i_image in enumerate(self.indexs_high_t_range):
                    image = self.temperatures_images[self.temperatures_images['i_image']==i_image]
                    # indexs_high_t_range selects top 30 images in a decreasing order
                    T_map_1tile[index,:,0] = image[['CROSSBAR_BTM0Q(K)']].values.T[0]
                    T_map_1tile[index,:,1] = image[['CROSSBAR_BTM1Q(K)']].values.T[0]
                    T_map_1tile[index,:,2] = image[['CROSSBAR_TOP2Q(K)']].values.T[0]
                    T_map_1tile[index,:,3] = image[['CROSSBAR_TOP3Q(K)']].values.T[0]
                # first compute the current of dummy matrix
                T_map_dummy_min = np.min(T_map_1tile, 2) # we first set the dummy matrix as minimum temperature of xbar_blocks
                # we take the first flatten feature of each image as the baseline dummy temperature, 
                # i.e. computations within an image share a common dummy matrix, i.e. temperature sensors update per image
                T_map_dummy_1d = T_map_dummy_min[:,0]
                # repeat and expand the array from dimention(n_images) to (n_images, n_flatten_features, n_blocks)
                T_map_dummy = np.expand_dims(np.expand_dims(T_map_dummy_1d,1).repeat(T_map_1tile.shape[1],axis=1),2).repeat(T_map_1tile.shape[2],axis=2)
                # call multiprocessors to compute the affected current
                pool = Pool()
                I_dummy_list = pool.map(I_V_T_smallSim.I_V_T_sim_fixedV, T_map_dummy.flatten())
                pool.close()
                pool.join()
                I_dummy_arr = np.array(I_dummy_list)
                print("\nI_dummy_arr.shape: ", I_dummy_arr.shape)
                print("weight.shape: ", weight.shape)
                I_dummy_ON_map = I_dummy_arr[:,0].reshape(input.shape[0],input.shape[2]*input.shape[3],self.numblock)
                I_dummy_OFF_map = I_dummy_arr[:,1].reshape(input.shape[0],input.shape[2]*input.shape[3],self.numblock)

                pool = Pool()
                I_partial_list = pool.map(I_V_T_smallSim.I_V_T_sim_fixedV, T_map_1tile.flatten())
                pool.close()
                pool.join()
                I_partial_arr = np.array(I_partial_list)
                I_partial_ON_map = I_partial_arr[:,0].reshape(input.shape[0],input.shape[2]*input.shape[3],self.numblock)
                I_partial_OFF_map = I_partial_arr[:,1].reshape(input.shape[0],input.shape[2]*input.shape[3],self.numblock)

            # generate impacts according to temperature maps
            # 1. dummy xbar array

            for i in range (self.weight.shape[2]):
                for j in range (self.weight.shape[3]):
                    # need to divide to different subArray
                    numSubArray = int(weight.shape[1]/self.subArray)
                    # cut into different subArrays
                    if numSubArray == 0:
                        mask = torch.zeros_like(weight)
                        mask[:,:,i,j] = 1
                        if weight.shape[1] == 3:
                            # after get the spacial kernel, need to transfer floating weight [-1, 1] to binarized ones
                            X_decimal = torch.round((2**bitWeight - 1)/2 * (weight+1) + 0)*mask
                            outputP = torch.zeros_like(output)
                            outputD = torch.zeros_like(output)
                            for k in range (int(bitWeight/self.cellBit)):
                                remainder = torch.fmod(X_decimal, cellRange)*mask
                                # retention
                                remainder = wage_quantizer.Retention(remainder,self.t,self.v,self.detect,self.target)
                                variation = np.random.normal(0, self.vari, list(weight.size())).astype(np.float32)
                                X_decimal = torch.round((X_decimal-remainder)/cellRange)*mask
                                # Now also consider weight has on/off ratio effects
                                # Here remainder is the weight mapped to Hardware, so we introduce on/off ratio in this value
                                # the range of remainder is [0, cellRange-1], we truncate it to [lower, upper]
                                remainderQ = (upper-lower)*(remainder-0)+(cellRange-1)*lower   # weight cannot map to 0, but to Gmin
                                remainderQ = remainderQ + remainderQ*torch.from_numpy(variation).cuda()
                                outputPartial= F.conv2d(input, remainderQ*mask, self.bias, self.stride, self.padding, self.dilation, self.groups)
                                outputDummyPartial= F.conv2d(input, dummyP*mask, self.bias, self.stride, self.padding, self.dilation, self.groups)
                                scaler = cellRange**k
                                outputP = outputP + outputPartial*scaler*2/(1-1/onoffratio)
                                outputD = outputD + outputDummyPartial*scaler*2/(1-1/onoffratio)
                            outputP = outputP - outputD
                            output = output + outputP
                        else:
                            # print("input.shape: ", input.shape)
                            # quantize input into binary sequence
                            inputQ = torch.round((2**bitActivation - 1)/1 * (input-0) + 0)
                            outputIN = torch.zeros_like(output)
                            # print("inputB.shape for F.Conv2d function: ", input.shape)
                            # print("output.shape from F.Conv2d function: ", output.shape)
                            for z in range(bitActivation):
                                inputB = torch.fmod(inputQ, 2)
                                inputQ = torch.round((inputQ-inputB)/2)
                                outputP = torch.zeros_like(output)
                                
                                # after get the spacial kernel, need to transfer floating weight [-1, 1] to binarized ones
                                X_decimal = torch.round((2**bitWeight - 1)/2 * (weight+1) + 0)*mask
                                outputD = torch.zeros_like(output)
                                for k in range (int(bitWeight/self.cellBit)):
                                    remainder = torch.fmod(X_decimal, cellRange)*mask
                                    # retention
                                    remainder = wage_quantizer.Retention(remainder,self.t,self.v,self.detect,self.target)
                                    variation = np.random.normal(0, self.vari, list(weight.size())).astype(np.float32)
                                    X_decimal = torch.round((X_decimal-remainder)/cellRange)*mask
                                    # Now also consider weight has on/off ratio effects
                                    # Here remainder is the weight mapped to Hardware, so we introduce on/off ratio in this value
                                    # the range of remainder is [0, cellRange-1], we truncate it to [lower, upper]
                                    remainderQ = (upper-lower)*(remainder-0)+(cellRange-1)*lower   # weight cannot map to 0, but to Gmin
                                    remainderQ = remainderQ + remainderQ*torch.from_numpy(variation).cuda()
                                    
                                    
                                    outputPartial= F.conv2d(inputB, remainderQ*mask, self.bias, self.stride, self.padding, self.dilation, self.groups)
                                    outputDummyPartial= F.conv2d(inputB, dummyP*mask, self.bias, self.stride, self.padding, self.dilation, self.groups)
                                    # Add ADC quanization effects here !!!
                                    outputPartialQ = wage_quantizer.LinearQuantizeOut(outputPartial, self.ADCprecision)
                                    outputDummyPartialQ = wage_quantizer.LinearQuantizeOut(outputDummyPartial, self.ADCprecision)
                                    scaler = cellRange**k
                                    outputP = outputP + outputPartialQ*scaler*2/(1-1/onoffratio)
                                    outputD = outputD + outputDummyPartialQ*scaler*2/(1-1/onoffratio)
                                scalerIN = 2**z
                                outputIN = outputIN + (outputP - outputD)*scalerIN
                            output = output + outputIN/(2**bitActivation)
                    else:
                        
                        # quantize input into binary sequence
                        inputQ = torch.round((2**bitActivation - 1)/1 * (input-0) + 0)
                        outputIN = torch.zeros_like(output)
                        for z in range(bitActivation):
                            inputB = torch.fmod(inputQ, 2)
                            inputQ = torch.round((inputQ-inputB)/2)
                            outputP = torch.zeros_like(output)
                            for s in range(numSubArray):
                                mask = torch.zeros_like(weight)
                                mask[:,(s*self.subArray):(s+1)*self.subArray, i, j] = 1
                                # after get the spacial kernel, need to transfer floating weight [-1, 1] to binarized ones
                                X_decimal = torch.round((2**bitWeight - 1)/2 * (weight+1) + 0)*mask
                                outputSP = torch.zeros_like(output)
                                outputD = torch.zeros_like(output)
                                for k in range (int(bitWeight/self.cellBit)):
                                    remainder = torch.fmod(X_decimal, cellRange)*mask
                                    # retention
                                    remainder = wage_quantizer.Retention(remainder,self.t,self.v,self.detect,self.target)
                                    variation = np.random.normal(0, self.vari, list(weight.size())).astype(np.float32)
                                    X_decimal = torch.round((X_decimal-remainder)/cellRange)*mask
                                    # Now also consider weight has on/off ratio effects
                                    # Here remainder is the weight mapped to Hardware, so we introduce on/off ratio in this value
                                    # the range of remainder is [0, cellRange-1], we truncate it to [lower, upper]*(cellRange-1)
                                    # to simulate temperature to weight matrix, this is used only for layer except Conv1
                                    
                                    remainderQ = (upper-lower)*(remainder-0)+(cellRange-1)*lower   # weight cannot map to 0, but to Gmin
                                    remainderQ = remainderQ + remainderQ*torch.from_numpy(variation).cuda()
                                    
                                    # modify to alike pytorX
                                    
                                    input_unfold = F.unfold(inputB, kernel_size=self.kernel_size,
                                    dilation=self.dilation, padding=self.padding,
                                    stride=self.stride)
                                    weight_flatten = (remainderQ*mask).view(self.out_channels, -1)
                                    dummy_flatten = (dummyP*mask).view(self.out_channels, -1)
                                    # 2.2. add paddings
                                    weight_padded = F.pad(weight_flatten, w_pad,
                                                        mode='constant', value=0)
                                    dummy_padded = F.pad(dummy_flatten, w_pad,
                                                        mode='constant', value=0)
                                    input_padded = F.pad(input_unfold, input_pad,
                                                        mode='constant', value=0)
                                    # 2.3. reshape to crxb size
                                    input_crxb = input_padded.view(input.shape[0], 1, self.crxb_row,
                                                                self.subArray, input_padded.shape[2])
                                    weight_crxb = weight_padded.view(self.crxb_col, self.subArray,
                                                                    self.crxb_row, self.subArray).transpose(1, 2)
                                    dummy_crxb = dummy_padded.view(self.crxb_col, self.subArray,
                                                                    self.crxb_row, self.subArray).transpose(1, 2)
                                    # G_crxb = self.w2g(weight_crxb)

                                    # if z==s==k==0:
                                    #     print("\n inputB.shape: ", inputB.shape, "flatten and unfold \ninput_unfold.shape: ", input_unfold.shape, " weight_flatten.shape: ", weight_flatten.shape, 
                                    #     " input_crxb.shape: ", input_crxb.shape, " weight_crxb.shape: ", weight_crxb.shape) #, " G_crxb.shape: ", G_crxb.shape)

                                    # Start decomposation
                                    output_crxb_standard = torch.matmul(weight_crxb, input_crxb)
                                    output_partial_crxb = torch.zeros_like(output_crxb_standard)
                                    output_dummy_crxb = torch.zeros_like(output_crxb_standard)
                                    if self.layer_Conv == 1:
                                        # print("self.layer_Conv = ", self.layer_Conv)
                                        # decompose matrix multiplication, need to declare that this is only on the layer Conv1.
                                        for in_0 in range(input_crxb.shape[0]): # in_0 is batch size of images
                                            # I_ON_impact_dummy = I_dummy_ON_map[in_0,0,0]
                                            # I_OFF_impact_dummy = I_dummy_OFF_map[in_0,0,0]

                                            I_ON_impact_dummy = 1
                                            I_OFF_impact_dummy = 0.1
                                            # dummy_crxb[:,:,:,:] *= (I_ON_impact_dummy+I_OFF_impact_dummy)
                                            dummy_crxb[:,:,:,:] *= 1
                                            for w_0 in range(weight_crxb.shape[0]): # w_0 is the first dimention of weight_crxb
                                                for in_4 in range(input_crxb.shape[4]): # 1024 input vectors for the layer Conv1
                                                    # replace the ION and IOF with I with temperature impacts 
                                                    # the PE under observation is filled with I_partial, other PEs is filled with I_dummy
                                                    I_ON_partial_blocks = I_partial_ON_map[in_0,in_4]
                                                    I_OFF_partial_blocks = I_partial_OFF_map[in_0,in_4]

                                                    for i_PE in range(weight_crxb.shape[1]):
                                                        if i_PE == 8 and self.temperature_sim == True:
                                                            for i_block in range(I_ON_partial_blocks.shape[0]):
                                                                weight_crxb[w_0,i_PE,self.blocksize*i_block:self.blocksize*(i_block+1),:] = \
                                                                    (I_ON_partial_blocks[i_block]-I_OFF_partial_blocks[i_block])\
                                                                    *(weight_crxb[w_0,i_PE,self.blocksize*i_block:self.blocksize*(i_block+1),:]-0)\
                                                                    +(cellRange-1)*I_OFF_partial_blocks[i_block]
                                                        else:
                                                            # weight_crxb[w_0,i_PE,:,:] = (I_ON_impact_dummy-I_OFF_impact_dummy)\
                                                            #     *(weight_crxb[w_0,i_PE,:,:]-0)+(cellRange-1)*I_OFF_impact_dummy
                                                            weight_crxb[w_0,i_PE,:,:] = (I_ON_impact_dummy)*(weight_crxb[w_0,i_PE,:,:]-0)
                                                        
                                                    # sub_input_unsqueeze = (input_crxb[i_0,i_1,:,:,i_4]).unsqueeze(2).unsqueeze(0).unsqueeze(0)
                                                    mul_partial_crxb = torch.matmul(weight_crxb[w_0,:,:,:], input_crxb[in_0,0,:,:,in_4].unsqueeze(2)).squeeze(2)# torch.matmul(weight_crxb, sub_input_unsqueeze)
                                                    mul_dummy_crxb = torch.matmul(dummy_crxb[w_0,:,:,:], input_crxb[in_0,0,:,:,in_4].unsqueeze(2)).squeeze(2)# torch.matmul(weight_crxb, sub_input_unsqueeze)
                                                    # output_partial_crxb[in_0,w_0,:,:,in_4] += mul_partial_crxb/I_ON_impact_dummy
                                                    # output_dummy_crxb[in_0,w_0,:,:,in_4] += mul_dummy_crxb/I_ON_impact_dummy

                                                    # move the divider to each image computation, which is supposed to greatly incerease the overhead.
                                                    output_partial_crxb[in_0,w_0,:,:,in_4] += mul_partial_crxb/I_ON_impact_dummy/(1-1/onoffratio)
                                                    output_dummy_crxb[in_0,w_0,:,:,in_4] += mul_dummy_crxb/I_ON_impact_dummy/(1-1/onoffratio)
                                        # check the difference between standard digitally computation and the flatten model
                                        # deviation_max = torch.max(output_crxb - output_crxb_standard).item()
                                        # deviation_min = torch.min(output_crxb - output_crxb_standard).item()
                                        # output_crxb_standard_max = torch.max(output_crxb_standard).item()
                                        # output_crxb_standard_min = torch.min(output_crxb_standard).item()
                                        # if abs(deviation_max) > 1:
                                        #     print("deviation of mul decompose: ", deviation_max, "min: ", deviation_min, " max of standard: ", output_crxb_standard_max, " min: ", output_crxb_standard_min)

                                        output_partial_sum = torch.sum(output_partial_crxb, dim=2)
                                        outputPartial = output_partial_sum.view(output_partial_sum.shape[0],
                                                                output_partial_sum.shape[1] * output_partial_sum.shape[2],
                                                                self.h_out,
                                                                self.w_out).index_select(dim=1, index=self.nchout_index)
                                        output_dummy_sum = torch.sum(output_dummy_crxb, dim=2)
                                        outputDummyPartial = output_dummy_sum.view(output_dummy_sum.shape[0],
                                                                output_dummy_sum.shape[1] * output_dummy_sum.shape[2],
                                                                self.h_out,
                                                                self.w_out).index_select(dim=1, index=self.nchout_index)
                                    else: # other layers is modeled in digital mode
                                        outputPartial= F.conv2d(inputB, remainderQ*mask, self.bias, self.stride, self.padding, self.dilation, self.groups)
                                        outputDummyPartial= F.conv2d(inputB, dummyP*mask, self.bias, self.stride, self.padding, self.dilation, self.groups)
                                    # outputPartial *= 2.8
                                    # end of pytorx convolutional computation
                                    
                                    # Add ADC quanization effects here !!!
                                    outputPartialQ = wage_quantizer.LinearQuantizeOut(outputPartial, self.ADCprecision)
                                    outputDummyPartialQ = wage_quantizer.LinearQuantizeOut(outputDummyPartial, self.ADCprecision)
                                    scaler = cellRange**k
                                    # outputSP = outputSP + outputPartialQ*scaler*2/(1-1/onoffratio)
                                    # outputD = outputD + outputDummyPartialQ*scaler*2/(1-1/onoffratio)
                                    outputSP = outputSP + outputPartialQ*scaler*2
                                    outputD = outputD + outputDummyPartialQ*scaler*2
                                # !!! Important !!! the dummy need to be multiplied by a ratio
                                outputSP = outputSP - outputD  # minus dummy column
                                # outputSP /= (upper) # here upper is the ON conductance, a small number in 1e-5 magnitude
                                outputP = outputP + outputSP
                            scalerIN = 2**z
                            outputIN = outputIN + outputP*scalerIN
                        output = output + outputIN/(2**bitActivation)
                        
            output = output/(2**bitWeight)   # since weight range was convert from [-1, 1] to [-256, 256]
            # output /= upper # here upper is the ON conductance, a small number in 1e-5 magnitude
            print("input.shape: ", input.shape, ", output.shape: ", output.shape)
            # if self.layer_Conv == 1:    # Conv layer 1 gets impacted most
                # print("output before and after the thermal imbalance impact: output_b = ", output, ", output_a = ", output*1.2)
                # output *= 400 
                # print(output.shape)
        elif self.inference == 1:
            weight1 = self.weight * self.scale + (self.weight - self.weight * self.scale).detach()
            weight = weight1 + (wage_quantizer.Q(weight1,self.wl_weight) -weight1).detach()
            weight = wage_quantizer.Retention(weight,self.t,self.v,self.detect,self.target)
            input = wage_quantizer.Q(input,self.wl_input)
            output= F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            output = wage_quantizer.LinearQuantizeOut(output, self.ADCprecision)
        else:
            # original WAGE QCov2d
            weight1 = self.weight * self.scale + (self.weight - self.weight * self.scale).detach()
            weight = weight1 + (wage_quantizer.Q(weight1,self.wl_weight) -weight1).detach()
            weight = wage_quantizer.Retention(weight,self.t,self.v,self.detect,self.target)
            output= F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        output = output/self.scale
        output = wage_quantizer.WAGEQuantizer_f(output, self.wl_activate, self.wl_error)
        
        return output


class QLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False,logger = None,clip_weight = False,wage_init=False,quantize_weight= False,clip_output =False,quantize_output = False,
	             wl_input =8,wl_activate=8,wl_error=8,wl_weight= 8,inference=0,onoffratio=10,cellBit=1,subArray=128,ADCprecision=5,vari=0,t=0,v=0,detect=0,target=0,debug = 0, name ='Qlinear', model = None):
        super(QLinear, self).__init__(in_features, out_features, bias)
        self.logger = logger
        self.clip_weight = clip_weight
        self.wage_init = wage_init
        self.quantize_weight = quantize_weight
        self.clip_output = clip_output
        self.debug = debug
        self.wl_weight = wl_weight
        self.quantize_output = quantize_output
        self.wl_activate = wl_activate
        self.wl_input = wl_input
        self.wl_error = wl_error
        self.inference = inference
        self.onoffratio = onoffratio
        self.cellBit = cellBit
        self.subArray = subArray
        self.ADCprecision = ADCprecision
        self.vari = vari
        self.t = t
        self.v = v
        self.detect = detect
        self.target = target
        self.name = name
        self.model = model
        self.scale  = wage_initializer.wage_init_(self.weight, self.wl_weight, factor=1.0)

    @weak_script_method
    def forward(self, input):

        weight1 = self.weight * self.scale + (self.weight - self.weight * self.scale).detach()
        weight = weight1 + (wage_quantizer.Q(weight1,self.wl_weight) -weight1).detach()
        outputOrignal = F.linear(input, weight, self.bias)
        output = torch.zeros_like(outputOrignal)

        bitWeight = int(self.wl_weight)
        bitActivation = int(self.wl_input)

        if self.inference == 1 and self.model=='VGG8':
            # set parameters for Hardware Inference
            onoffratio = self.onoffratio
            upper = 1
            lower = 1/onoffratio
            output = torch.zeros_like(outputOrignal)
            cellRange = 2**self.cellBit   # cell precision is 4
            # Now consider on/off ratio
            dummyP = torch.zeros_like(weight)
            dummyP[:,:] = (cellRange-1)*(upper+lower)/2
            # need to divide to different subArray
            numSubArray = int(weight.shape[1]/self.subArray)

            if numSubArray == 0:
                mask = torch.zeros_like(weight)
                mask[:,:] = 1
                # quantize input into binary sequence
                inputQ = torch.round((2**bitActivation - 1)/1 * (input-0) + 0)
                outputIN = torch.zeros_like(outputOrignal)
                for z in range(bitActivation):
                    inputB = torch.fmod(inputQ, 2)
                    inputQ = torch.round((inputQ-inputB)/2)
                    # after get the spacial kernel, need to transfer floating weight [-1, 1] to binarized ones
                    X_decimal = torch.round((2**bitWeight - 1)/2 * (weight+1) + 0)*mask
                    outputP = torch.zeros_like(outputOrignal)
                    outputD = torch.zeros_like(outputOrignal)
                    for k in range (int(bitWeight/self.cellBit)):
                        remainder = torch.fmod(X_decimal, cellRange)*mask
                        # retention
                        remainder = wage_quantizer.Retention(remainder,self.t,self.v,self.detect,self.target)
                        variation = np.random.normal(0, self.vari, list(weight.size())).astype(np.float32)
                        X_decimal = torch.round((X_decimal-remainder)/cellRange)*mask
                        # Now also consider weight has on/off ratio effects
                        # Here remainder is the weight mapped to Hardware, so we introduce on/off ratio in this value
                        # the range of remainder is [0, cellRange-1], we truncate it to [lower, upper]
                        remainderQ = (upper-lower)*(remainder-0)+(cellRange-1)*lower   # weight cannot map to 0, but to Gmin
                        remainderQ = remainderQ + remainderQ*torch.from_numpy(variation).cuda()
                        outputPartial= F.linear(inputB, remainderQ*mask, self.bias)
                        outputDummyPartial= F.linear(inputB, dummyP*mask, self.bias)
                        # Add ADC quanization effects here !!!
                        outputPartialQ = wage_quantizer.LinearQuantizeOut(outputPartial, self.ADCprecision)
                        outputDummyPartialQ = wage_quantizer.LinearQuantizeOut(outputDummyPartial, self.ADCprecision)
                        scaler = cellRange**k
                        outputP = outputP + outputPartialQ*scaler*2/(1-1/onoffratio)
                        outputD = outputD + outputDummyPartialQ*scaler*2/(1-1/onoffratio)
                    scalerIN = 2**z
                    outputIN = outputIN + (outputP - outputD)*scalerIN
                output = output + outputIN/(2**bitActivation)
            else:
                inputQ = torch.round((2**bitActivation - 1)/1 * (input-0) + 0)
                outputIN = torch.zeros_like(outputOrignal)
                for z in range(bitActivation):
                    inputB = torch.fmod(inputQ, 2)
                    inputQ = torch.round((inputQ-inputB)/2)
                    outputP = torch.zeros_like(outputOrignal)
                    for s in range(numSubArray):
                        mask = torch.zeros_like(weight)
                        mask[:,(s*self.subArray):(s+1)*self.subArray] = 1
                        # after get the spacial kernel, need to transfer floating weight [-1, 1] to binarized ones
                        X_decimal = torch.round((2**bitWeight - 1)/2 * (weight+1) + 0)*mask
                        outputSP = torch.zeros_like(outputOrignal)
                        outputD = torch.zeros_like(outputOrignal)
                        for k in range (int(bitWeight/self.cellBit)):
                            remainder = torch.fmod(X_decimal, cellRange)*mask
                            # retention
                            remainder = wage_quantizer.Retention(remainder,self.t,self.v,self.detect,self.target)
                            variation = np.random.normal(0, self.vari, list(remainder.size())).astype(np.float32)
                            X_decimal = torch.round((X_decimal-remainder)/cellRange)*mask
                            # Now also consider weight has on/off ratio effects
                            # Here remainder is the weight mapped to Hardware, so we introduce on/off ratio in this value
                            # the range of remainder is [0, cellRange-1], we truncate it to [lower, upper]*(cellRange-1)
                            remainderQ = (upper-lower)*(remainder-0)+(cellRange-1)*lower   # weight cannot map to 0, but to Gmin
                            remainderQ = remainderQ + remainderQ*torch.from_numpy(variation).cuda()
                            outputPartial= F.linear(inputB, remainderQ*mask, self.bias)
                            outputDummyPartial= F.linear(inputB, dummyP*mask, self.bias)
                            # Add ADC quanization effects here !!!
                            outputPartialQ = wage_quantizer.LinearQuantizeOut(outputPartial, self.ADCprecision)
                            outputDummyPartialQ = wage_quantizer.LinearQuantizeOut(outputDummyPartial, self.ADCprecision)
                            scaler = cellRange**k
                            outputSP = outputSP + outputPartialQ*scaler*2/(1-1/onoffratio)
                            outputD = outputD + outputDummyPartialQ*scaler*2/(1-1/onoffratio)
                        outputSP = outputSP - outputD  # minus dummy column
                        outputP = outputP + outputSP
                    scalerIN = 2**z
                    outputIN = outputIN + outputP*scalerIN
                output = output + outputIN/(2**bitActivation)
            output = output/(2**bitWeight)
        
        elif self.inference == 1:
            weight1 = self.weight * self.scale + (self.weight - self.weight * self.scale).detach()
            weight = weight1 + (wage_quantizer.Q(weight1,self.wl_weight) -weight1).detach()
            weight = wage_quantizer.Retention(weight,self.t,self.v,self.detect,self.target)
            input = wage_quantizer.Q(input,self.wl_input)
            output= F.linear(input, weight, self.bias)
            output = wage_quantizer.LinearQuantizeOut(output, self.ADCprecision)
        else:
            # original WAGE QCov2d
            weight1 = self.weight * self.scale + (self.weight - self.weight * self.scale).detach()
            weight = weight1 + (wage_quantizer.Q(weight1,self.wl_weight) -weight1).detach()
            weight = wage_quantizer.Retention(weight,self.t,self.v,self.detect,self.target)
            output = F.linear(input, weight, self.bias)
        
        output = output/self.scale
        output = wage_quantizer.WAGEQuantizer_f(output,self.wl_activate, self.wl_error)
        
        return output

