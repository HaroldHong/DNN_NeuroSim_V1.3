from utee import wage_initializer,wage_quantizer, I_V_T_smallSim

fig189_pe6 = [345.539,347.076,346.823,344.961]

for T in fig189_pe6:
    I_ON,I_OFF = I_V_T_smallSim.I_V_T_sim_fixedV(T)

    print("T: ",T, " I_ON: ",I_ON,"I_OFF: ",I_OFF,"I_ON/I_OFF: ",I_ON/I_OFF)