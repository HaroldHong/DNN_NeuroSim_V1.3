import matplotlib.pyplot as plt
import numpy as np
import math
import random
import csv
import pandas as pd
from scipy.special import erfinv
from scipy.special.basic import _range_prod
from multiprocessing import Pool

numColPerSynapse = 4
NormalizedMin = 0
NormalizedMax = pow(2, numColPerSynapse)

RealMax = 1
RealMin = -1
q = 1.6e-19; h = 4.135667e-15; k = 8.61733034e-5; rou_w = 3e-2; N=5; f = 0.1 # h is Planck constant in eVs, k is Boltzmann constant in eV*K^-1
def I_V_T_sim(V, T):
    
    alpha = 2.80; phi_0 = 1.14; gamma = 0.95; theta = 7.3e-4; T0 = 300; V0=1e-9 # parameters for OFF-state without cycling,
    phi_V0_T0 = phi_0 - gamma*V0 - theta*T0; phi_V = phi_0 - gamma*V; phi_V_T = phi_0 - gamma*V - theta*T; G_0 = 1/12.9*1e-3
    R_OFF_target = 10/G_0
    # model of 2014' JAP "Multi-scale quantum point..."
    tB_0 = 0.25e-9; t_0 = 0.12e-9 # RESET(LRS->HRS): 0.25, SET(HRS->LRS): 0.59
    m_0 = 9.11e-31*5.6096e35; m_star = 1.08*m_0
    alpha1 = tB_0*math.pi*math.pi/h*math.sqrt(2*m_star/phi_0)
    alpha2 =  tB_0/(t_0*phi_V)
    alpha3 = math.log(1/(R_OFF_target*N*G_0))/(-phi_0)
    alpha = alpha
    # print("alpha1: ", alpha1,", alpha2: ", alpha2, ", alpha3: ", alpha3)

    # I_OFF = 2.0*q/(h*alpha)*math.exp(-alpha*phi_V_T)/np.sinc(alpha*k*T)*(1-math.exp(-alpha*V))
    I_OFF = 2.0*q/(h*alpha)*math.exp(-alpha*phi_V_T)*(1-math.exp(-alpha*V))
    I_LHRS = 2*q/h*N*(V+1/alpha*math.log((1+math.exp(alpha*(phi_0-0.9*V)))/(1+math.exp(alpha*(phi_0+(1-0.9)*V)))))
    RON_0 =  13e3
    R_ON_base = RON_0*(1+rou_w*(T-T0))
    I_ON_base = N*G_0/(1+N*G_0*R_ON_base)*V
    # print(math.log((1+math.exp(alpha*(phi_0-gamma*V)))/(1+math.exp(alpha*(phi_0+(1-gamma)*V)))))
    # print(I_OFF, V/I_OFF)
    # print("I_OFF: ", I_OFF, "I_ON_base: ", I_ON_base, "V/I_OFF: ", V/I_OFF, "V/I_ON_base: ", V/I_ON_base, "I_ON_base/I_OFF: ", I_ON_base/I_OFF)
    # print(V/I_LHRS)
    # print(1/(N*G_0*math.exp(-alpha*phi_V)))
    return I_ON_base, I_OFF

def I_V_T_sim_fixedV(T):
    V = 0.2
    # alpha = 2.80; phi_0 = 1.14; 
    # alpha = 5.11; phi_0 = 0.77; 
    alpha = 6.55; phi_0 = 0.67; 
    gamma = 0.95; theta = 7.3e-4; T0 = 293; V0=1e-9 # parameters for OFF-state without cycling,
    
    phi_V0_T0 = phi_0 - gamma*V0 - theta*T0; phi_V = phi_0 - gamma*V; phi_V_T = phi_0 - gamma*V - theta*T; G_0 = 1/12.9*1e-3
    R_OFF_target = 10/G_0
    # model of 2014' JAP "Multi-scale quantum point..."
    tB_0 = 0.25e-9; t_0 = 0.12e-9 # RESET(LRS->HRS): 0.25, SET(HRS->LRS): 0.59
    m_0 = 9.11e-31*5.6096e35; m_star = 1.08*m_0
    alpha1 = tB_0*math.pi*math.pi/h*math.sqrt(2*m_star/phi_0)
    alpha2 =  tB_0/(t_0*phi_V)
    alpha3 = math.log(1/(R_OFF_target*N*G_0))/(-phi_0)
    alpha = alpha
    # print("alpha1: ", alpha1,", alpha2: ", alpha2, ", alpha3: ", alpha3)

    # I_OFF = 2.0*q/(h*alpha)*math.exp(-alpha*phi_V_T)/np.sinc(alpha*k*T)*(1-math.exp(-alpha*V))
    I_OFF = 2.0*q/(h*alpha)*math.exp(-alpha*phi_V_T)*(1-math.exp(-alpha*V))
    N_LHRS = 3
    I_LHRS = 2*q/h*N_LHRS*(V+1/alpha*math.log((1+math.exp(alpha*(phi_0-0.9*V)))/(1+math.exp(alpha*(phi_0+(1-0.9)*V)))))
    RON_0 =  13e3
    R_ON_base = RON_0*(1+rou_w*(T-T0))
    I_ON_base = N*G_0/(1+N*G_0*R_ON_base)*V
    
    return I_ON_base, I_OFF

def getConductanceRow_Neurosim(weight_4bit): # , maxConductance, minConductance):
    cellrange = pow(2, 1)
    newdata = ((NormalizedMax-NormalizedMin)/(RealMax-RealMin)*(weight_4bit-RealMax)+NormalizedMax)
    value = int(newdata)
    vec_synapse = []; weightrow=[]
    vec_synapse.insert(0, int(weight_4bit<0))
    if weight_4bit>=0:
        value = value-pow(2,3)
    for i in range(3):
        v_bit = int(value >= pow(2,2-i))
        value = value-v_bit*pow(2,2-i)
        vec_synapse.append(v_bit)

    return vec_synapse

def frac_bin_to_decimal(synapse_vec):
    
    f_convert = 0; sign_bit = 1
    newdata_unsigned = 0

    if synapse_vec[0] == 1:
        sign_bit *= -1
    
    for i_bit in range(1,numColPerSynapse):
        newdata_unsigned += synapse_vec[i_bit]*pow(2, numColPerSynapse-1-i_bit)
    if sign_bit == 1:
        newdata_unsigned += pow(2, numColPerSynapse-1)

    f_convert = (newdata_unsigned-NormalizedMax)*(RealMax-RealMin)/(NormalizedMax-NormalizedMin)+RealMax
    # print(synapse_vec, f_convert)
    return f_convert

Vread = 0.2; T1 = 330; T2 = 327; T0 = 300; numRow = 4; numCol = 4

def frac_sum_bin_to_decimal(sum_vec):
    sum_dec = 0
    
    for i in range(numColPerSynapse):
        unit_vec = np.zeros(numColPerSynapse)
        unit_vec[i] = 1
        sum_dec += math.floor(sum_vec[i])*frac_bin_to_decimal(unit_vec)
    # print(sum_vec, sum_dec)

    return sum_dec
def boxmullersampling(mu=0, sigma=1, size=1):
    u = np.random.uniform(size=size)
    v = np.random.uniform(size=size)
    z = np.sqrt(-2*np.log(u))*np.cos(2*np.pi*v)
    return mu+z*sigma

# This function firstly converts the original fractional decimals into binary vectors in 2's complementary way. 
# Secondly, sum these vectors up and convert the sum-up vector into a fractional decimal back.
# Here we simulate 3 scenarios 1) digital(ideal), 2) with the impact of temperature, 3) plus normal deviation based on 2)#
def MAC_with_impact(pairs_weight_temperature):
    vec_current_accumulate_with_deviation = np.zeros(4);vec_current_accumulate_wo_deviation = np.zeros(4)
    vec_multiply_accumulate = np.zeros(4)
    numRow = len(pairs_weight_temperature)
    for i_row in range(numRow):
        vec_synapse = getConductanceRow_Neurosim(pairs_weight_temperature[i_row][0])
        T = pairs_weight_temperature[i_row][1]
        # print(T)
        I_ON_mean, I_OFF_mean = I_V_T_sim(Vread, T)
        I_rms_deviation_OFF = 0.1 * I_OFF_mean
        I_rms_deviation_ON = 0.1 * I_ON_mean
        # I_rms_deviation_OFF = math.sqrt(I_OFF_mean/Vread*f*1e9*q*(4*k*T+2*Vread))
        # I_rms_deviation_ON = math.sqrt(I_ON_mean/Vread*f*1e9*q*(4*k*T+2*Vread))
        for i_col in range(numCol):
            if vec_synapse[i_col] == 1:
                currentRead_with_deviation = I_ON_mean+boxmullersampling(mu=0, sigma=I_rms_deviation_ON, size=1)[0]
                currentRead_wo_deviation = I_ON_mean
            else:
                currentRead_with_deviation = I_OFF_mean+boxmullersampling(mu=0, sigma=I_rms_deviation_OFF, size=1)[0]
                currentRead_wo_deviation = I_OFF_mean
            vec_multiply_accumulate[i_col] += vec_synapse[i_col]
            vec_current_accumulate_with_deviation[i_col] += currentRead_with_deviation
            vec_current_accumulate_wo_deviation[i_col] += currentRead_wo_deviation

    return vec_multiply_accumulate, vec_current_accumulate_with_deviation, vec_current_accumulate_wo_deviation
# maxConductance = Vread/I_OFF; minConductance = Vread/I_ON

# pairs_weight_temperature_T0 = []
# pairs_weight_temperature_T0.append([0.75,T0]); pairs_weight_temperature_T0.append([0.125,T0]); 
# pairs_weight_temperature_T0.append([0.625,T0]); pairs_weight_temperature_T0.append([0.875,T0]); 

# pairs_weight_temperature_T1 = []
# pairs_weight_temperature_T1.append([0.75,T1]); pairs_weight_temperature_T1.append([0.125,T1]); 
# pairs_weight_temperature_T1.append([0.625,T1]); pairs_weight_temperature_T1.append([0.875,T1]); 

# pairs_weight_temperature_T2 = []; 
# pairs_weight_temperature_T2.append([0.75,T2]); pairs_weight_temperature_T2.append([0.125,T2]); 
# pairs_weight_temperature_T2.append([0.625,T2]); pairs_weight_temperature_T2.append([0.875,T2]); 

# vec_multiply_accumulate_T1, vec_current_accumulate_T1 = MAC_with_impact(pairs_weight_temperature_T1)
# vec_multiply_accumulate_T2, vec_current_accumulate_T2 = MAC_with_impact(pairs_weight_temperature_T2)
# vec_multiply_accumulate_T0, vec_current_accumulate_T0 = MAC_with_impact(pairs_weight_temperature_T0)

# frac_sum_bin_to_decimal(vec_multiply_accumulate_T1)
# frac_sum_bin_to_decimal(vec_multiply_accumulate_T1*6)
# frac_sum_bin_to_decimal(vec_current_accumulate_T1*6*1e5)

# frac_sum_bin_to_decimal(vec_multiply_accumulate_T2)
# frac_sum_bin_to_decimal(vec_multiply_accumulate_T2*6)
# frac_sum_bin_to_decimal(vec_current_accumulate_T2*6*1e5)

# frac_sum_bin_to_decimal(vec_multiply_accumulate_T0)
# frac_sum_bin_to_decimal(vec_multiply_accumulate_T0*6)
# frac_sum_bin_to_decimal(vec_current_accumulate_T0*6*1e5)

# path_weight = "weightConv1_.csv"

# with open(path_weight) as f_weight:
#     w_csv = csv.reader(f_weight)
#     w_matrix_full = list(w_csv)
#     print(w_matrix_full[0])
#     for col in range(len(w_matrix_full[0])):
#         for row in range(len(w_matrix_full)):
#             w_matrix_full[row][col] = float(w_matrix_full[row][col].strip())
#     print(w_matrix_full[0])
#     pe_selected = 7; num_activerows = 30; blocksize=32
#     w_matrix_pe = w_matrix_full[pe_selected*128:(pe_selected+1)*128]

#     print(len(w_matrix_pe[0]))

#     T_low = random.randrange(310, 350, 1)
#     T_blocks = [T_low, T_low+1, T_low+2, T_low+1]
#     active_rows_rdn = random.sample(range(128), num_activerows)
#     print(active_rows_rdn)

#     list_MAC_base = []; list_MAC_impact = []

#     for i_weight in range(128):        
#         # i_weight_start = i_weight*4; i_weight_end = i_weight_start+4
#         pairs_weight_temperature_T = []
#         for i_active_row in range(num_activerows):
#             T_activerow = T_blocks[active_rows_rdn[i_active_row]//blocksize]
#             T_base = 300
#             pairs_weight_temperature_T.append([w_matrix_pe[i_active_row][i_weight], T_base])

#         vec_multiply_accumulate_T, vec_current_accumulate_T = MAC_with_impact(pairs_weight_temperature_T)
#         MAC_dec_base = frac_sum_bin_to_decimal(vec_multiply_accumulate_T)
#         MAC_dec_impact = frac_sum_bin_to_decimal(vec_current_accumulate_T*6*1e5)
#         list_MAC_base.append(MAC_dec_base)
#         list_MAC_impact.append(MAC_dec_impact)
#     nparr_MAC_base = np.asarray(list_MAC_base); nparr_MAC_impact = np.asarray(list_MAC_impact)
#     diff_MAC_base_impact = nparr_MAC_impact-nparr_MAC_base; ratio_diff_MAC_base_impact = diff_MAC_base_impact/(nparr_MAC_base+0.01)
#         # random.randrange(len(w_matrix[0])-4)
# print(nparr_MAC_base)
# print(nparr_MAC_impact)
# print(diff_MAC_base_impact)
# print(ratio_diff_MAC_base_impact)

# update

# path_weight = "weightConv1_.csv"
# path_result = "ConvertResult.csv"

# ConvertResult = []
# f_accurate_vec = []; f_dummy_wo_deviation_balance_vec = []; f_dummy_wo_deviation_unbalance_vec = []
# f_dummy_with_deviation_balance_vec = []; f_dummy_with_deviation_unbalance_vec = []
# balance_diff_with_deviation_vec = []; unbalance_diff_with_deviation_vec = []; relative_diff_with_deviation_vec = []
# balance_diff_wo_deviation_vec = []; unbalance_diff_wo_deviation_vec = []; relative_diff_wo_deviation_vec = []

# with open(path_weight) as f_weight:
#     w_csv = csv.reader(f_weight)
#     w_matrix_full = list(w_csv)
#     print(w_matrix_full[0])
#     for col in range(len(w_matrix_full[0])):
#         for row in range(len(w_matrix_full)):
#             w_matrix_full[row][col] = float(w_matrix_full[row][col].strip())
#     print(w_matrix_full[0])
#     pe_selected = 7; num_activerows = 40; blocksize=32
#     w_matrix_pe = w_matrix_full[pe_selected*128:(pe_selected+1)*128]

#     print(len(w_matrix_pe[0]))

#     T_low = random.randrange(3210, 3810, 1)/10
#     T_blocks = [T_low, T_low+0.3, T_low+2.5, T_low+0.5]
#     active_rows_rdn = random.sample(range(128), num_activerows)
#     print(active_rows_rdn)
#     list_MAC_base = []; list_MAC_impact = []
#     T_base = 300+math.floor((T_low-300)/5+0.5)*5
#     # T_base = T_low
#     I_ON_base, I_OFF_base = I_V_T_sim(0.2, T_base)
#     for i_weight in range(128):        
#         # i_weight_start = i_weight*4; i_weight_end = i_weight_start+4
#         pairs_weight_temperature_T_unbalance = []; pairs_weight_temperature_T_balance = []
#         for i_active_row in range(num_activerows):
#             T_activerow = T_blocks[active_rows_rdn[i_active_row]//blocksize]
#             pairs_weight_temperature_T_unbalance.append([w_matrix_pe[i_active_row][i_weight], T_activerow])
#             pairs_weight_temperature_T_balance.append([w_matrix_pe[i_active_row][i_weight], T_low])

#         vec_multiply_accumulate_T_balance, vec_current_accumulate_with_deviation_T_balance, vec_current_accumulate_wo_deviation_T_balance = MAC_with_impact(pairs_weight_temperature_T_balance)
#         vec_multiply_accumulate_T_unbalance, vec_current_accumulate_with_deviation_T_unbalance, vec_current_accumulate_wo_deviation_T_unbalance = MAC_with_impact(pairs_weight_temperature_T_unbalance)
#         f_accurate = frac_sum_bin_to_decimal(vec_multiply_accumulate_T_balance)
#         # f_accurate = frac_sum_bin_to_decimal(vec_multiply_accumulate_T_unbalance)
#         vec_current_minus_dummy_with_deviation_T_balance = vec_current_accumulate_with_deviation_T_balance - num_activerows*(I_OFF_base+I_ON_base)/2
#         vec_current_minus_dummy_wo_deviation_T_balance = vec_current_accumulate_wo_deviation_T_balance - num_activerows*(I_OFF_base+I_ON_base)/2
        
#         vec_current_minus_dummy_with_deviation_T_unbalance = vec_current_accumulate_with_deviation_T_unbalance - num_activerows*(I_OFF_base+I_ON_base)/2
#         vec_current_minus_dummy_wo_deviation_T_unbalance = vec_current_accumulate_wo_deviation_T_unbalance - num_activerows*(I_OFF_base+I_ON_base)/2
#         # print(vec_current_minus_dummy_T)
#         vec_minus_dummy_with_deviation_balance = [math.ceil(w / ((I_ON_base-I_OFF_base)/2)) for w in vec_current_minus_dummy_with_deviation_T_balance]
#         vec_minus_dummy_wo_deviation_balance = [math.ceil(w / ((I_ON_base-I_OFF_base)/2)) for w in vec_current_minus_dummy_wo_deviation_T_balance]
        
#         vec_minus_dummy_with_deviation_unbalance = [math.ceil(w / ((I_ON_base-I_OFF_base)/2)) for w in vec_current_minus_dummy_with_deviation_T_unbalance]
#         vec_minus_dummy_wo_deviation_unbalance = [math.ceil(w / ((I_ON_base-I_OFF_base)/2)) for w in vec_current_minus_dummy_wo_deviation_T_unbalance]
#         # print(vec_minus_dummy)
#         vec_convert_dummy_with_deviation_balance = np.asarray(vec_minus_dummy_with_deviation_balance)+num_activerows*np.ones(numColPerSynapse)
#         vec_convert_dummy_with_deviation_balance = vec_convert_dummy_with_deviation_balance//2

#         vec_convert_dummy_wo_deviation_balance = np.asarray(vec_minus_dummy_wo_deviation_balance)+num_activerows*np.ones(numColPerSynapse)
#         vec_convert_dummy_wo_deviation_balance = vec_convert_dummy_wo_deviation_balance//2

#         vec_convert_dummy_with_deviation_unbalance = np.asarray(vec_minus_dummy_with_deviation_unbalance)+num_activerows*np.ones(numColPerSynapse)
#         vec_convert_dummy_with_deviation_unbalance = vec_convert_dummy_with_deviation_unbalance//2

#         vec_convert_dummy_wo_deviation_unbalance = np.asarray(vec_minus_dummy_wo_deviation_unbalance)+num_activerows*np.ones(numColPerSynapse)
#         vec_convert_dummy_wo_deviation_unbalance = vec_convert_dummy_wo_deviation_unbalance//2
#         # print(vec_convert_dummy//2)
#         f_dummy_with_deviation_balance = frac_sum_bin_to_decimal(vec_convert_dummy_with_deviation_balance)
#         f_dummy_wo_deviation_balance = frac_sum_bin_to_decimal(vec_convert_dummy_wo_deviation_balance)
        
#         f_dummy_with_deviation_unbalance = frac_sum_bin_to_decimal(vec_convert_dummy_with_deviation_unbalance)
#         f_dummy_wo_deviation_unbalance = frac_sum_bin_to_decimal(vec_convert_dummy_wo_deviation_unbalance)
#         # print("without deviation: f_accurate, f_dummy_wo_deviation_balance, f_dummy_wo_deviation_unbalance, balance difference(%), unbalance difference(%), relative difference(%)")
#         # print(f_accurate, f_dummy_wo_deviation_balance, f_dummy_wo_deviation_unbalance, ' {}%, {}%, {}%'.format((f_accurate-f_dummy_wo_deviation_balance)/f_accurate*100,
#         #  (f_accurate-f_dummy_wo_deviation_unbalance)/f_accurate*100, (f_accurate-f_dummy_wo_deviation_unbalance)/f_accurate*100-(f_accurate-f_dummy_wo_deviation_balance)/f_accurate*100))
        
#         # print("with deviation: f_accurate, f_dummy_with_deviation_balance, f_dummy_with_deviation_unbalance, balance difference(%), unbalance difference(%), relative difference(%)")
#         # print(f_accurate, f_dummy_with_deviation_balance, f_dummy_with_deviation_unbalance, ' {}%, {}%, {}%'.format((f_accurate-f_dummy_with_deviation_balance)/f_accurate*100,
#         #  (f_accurate-f_dummy_with_deviation_unbalance)/f_accurate*100, (f_accurate-f_dummy_with_deviation_unbalance)/f_accurate*100-(f_accurate-f_dummy_with_deviation_balance)/f_accurate*100))
        


#         f_accurate_vec.append(f_accurate); f_dummy_wo_deviation_balance_vec.append(f_dummy_wo_deviation_balance)
#         f_dummy_wo_deviation_unbalance_vec.append(f_dummy_wo_deviation_unbalance); f_dummy_with_deviation_balance_vec.append(f_dummy_with_deviation_balance)
#         f_dummy_with_deviation_unbalance_vec.append(f_dummy_with_deviation_unbalance)

#         balance_diff_wo_deviation_vec.append(abs((f_accurate-f_dummy_wo_deviation_balance)/(f_accurate+1e-10)))
#         unbalance_diff_wo_deviation_vec.append(abs((f_accurate-f_dummy_wo_deviation_unbalance)/(f_accurate+1e-10)))
#         relative_diff_wo_deviation_vec.append(unbalance_diff_wo_deviation_vec[-1]-balance_diff_wo_deviation_vec[-1])

#         balance_diff_with_deviation_vec.append(abs((f_accurate-f_dummy_with_deviation_balance)/(f_accurate+1e-10)))
#         unbalance_diff_with_deviation_vec.append(abs((f_accurate-f_dummy_with_deviation_unbalance)/(f_accurate+1e-10)))
#         relative_diff_with_deviation_vec.append(unbalance_diff_with_deviation_vec[-1]-balance_diff_with_deviation_vec[-1])

#         result_zip = zip(f_accurate_vec,f_dummy_wo_deviation_balance_vec,f_dummy_wo_deviation_unbalance_vec,
#         balance_diff_wo_deviation_vec,unbalance_diff_wo_deviation_vec,relative_diff_wo_deviation_vec,
#         f_accurate_vec,f_dummy_with_deviation_balance_vec,f_dummy_with_deviation_unbalance_vec,
#         balance_diff_with_deviation_vec,unbalance_diff_with_deviation_vec,relative_diff_with_deviation_vec)

#         name = ['f_accurate_vec','f_dummy_wo_deviation_balance_vec','f_dummy_wo_deviation_unbalance_vec',
#         'balance_diff_wo_deviation_vec','unbalance_diff_wo_deviation_vec','relative_diff_wo_deviation_vec',
#         'f_accurate_vec','f_dummy_with_deviation_balance_vec','f_dummy_with_deviation_unbalance_vec',
#         'balance_diff_with_deviation_vec','unbalance_diff_with_deviation_vec','relative_diff_with_deviation_vec']
#         result_table = pd.DataFrame(columns=name,data=result_zip)
#         result_table.to_csv('./MAC_Temperature_Impact.csv',encoding='utf8')
#     print(T_base, T_low)
    

# f_test = [-0.375,0.125,0.75,0.375,-0.875,-0.25,0.75,-1,0.5,-0.5,-0.625,0.125,0.625,0.25,-0.5,0.875,-0.375,0.125,-0.625,0.25,-0.75,0.25,-0.125,0.875,0.875,0.375,0.125,0.5,-0.375,-0.75]
# f_test = []; acc_sum = 0
# for i in range(20):
#     test = random.random()*2-1; acc_sum = acc_sum+test
#     f_test.append(test)
# print(sum(f_test), len(f_test))
# pairs_weight_temperature_T300 = []
# for i in range(len(f_test)):
#     pairs_weight_temperature_T300.append([f_test[i], 300])
# print(pairs_weight_temperature_T300)
# # vec_multiply_accumulate_T300, vec_current_accumulate_T300
# vec_multiply_accumulate_T300, vec_current_accumulate_T300_with_deviation, vec_current_accumulate_T300_wo_deviation = MAC_with_impact(pairs_weight_temperature_T300)
# f_accurate = frac_sum_bin_to_decimal(vec_multiply_accumulate_T300)
# f_reconstruct = frac_sum_bin_to_decimal(vec_current_accumulate_T300_with_deviation*6*1e5)

# # test python parallization

# T_test = np.zeros((3,4))+300

# pool = Pool()
# I_list = pool.map(I_V_T_sim_fixedV, T_test.flatten())
# pool.close()
# pool.join()
# I_arr = np.array(I_list)
# I_ON_arr = I_arr[:,0]
# I_OFF_arr = I_arr[:,1]
# I_ON, I_OFF = I_V_T_sim(0.2, 300)
# print("I_ON_arr, I_OFF_arr ")
# print( I_ON_arr )
# print(I_OFF_arr)
# # print('\n', I_ON_arr, I_OFF_arr, (I_OFF_arr+I_ON_arr)/2, (I_ON_arr-I_OFF_arr)/2)

# print('\n',vec_multiply_accumulate_T300, vec_current_accumulate_T300_with_deviation)


# vec_current_minus_dummy_T300 = vec_current_accumulate_T300_with_deviation - len(f_test)*(I_OFF+I_ON)/2
# print('\n',vec_current_minus_dummy_T300)
# vec_minus_dummy = [math.ceil(w / ((I_ON-I_OFF)/2)) for w in vec_current_minus_dummy_T300]
# print('\n',vec_minus_dummy)
# min_negative = 0
# if min(vec_minus_dummy)<0:
#     min_negative = min(vec_minus_dummy)
# vec_convert_dummy = np.asarray(vec_minus_dummy)+len(f_test)*np.ones(numColPerSynapse)
# vec_convert_dummy = vec_convert_dummy//2
# print('\n',vec_convert_dummy//2)
# f_dummy = frac_sum_bin_to_decimal(vec_convert_dummy)
# print('\n',acc_sum, f_accurate, f_dummy)
