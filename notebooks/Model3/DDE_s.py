import numpy as np
from copy import deepcopy
from Parameters_s import *
import sympy
from jitcdde import y, t, jitcdde


sympy_params = {k: sympy.symbols(k) for k in params.keys()}
sympy_delays = {k: sympy.symbols(k) for k in T_delay.keys()}

y_sym = {k: y(v) for k,v in states.items()} # 현재 시간의 상태
y_delay_sym = {k: (lambda state_key, delay_key: y(states[state_key], t - sympy_delays[delay_key])) for k in states.keys()}

f = [0] * num_states

# Infected (현재 시간의 y_sym 사용)
Infected = sympy_params['r'] * (y_sym['PS'] + y_sym['PA'] + y_sym['IA'] + y_sym['ATN']) + y_sym['IS'] + y_sym['STN']

# Susceptible: S
f[states['S']] = (y_sym['NTN'] + y_sym['GTN'] + (1 - (sympy_params['phi_a_1'] + sympy_params['phi_a_2'])) * y_delay_sym['F_FPS']('F_FPS', 'gamma')
                  + y_sym['FTN'] + (1 - (sympy_params['phi_s_1'] + sympy_params['phi_s_2'])) * (y_sym['SF'] + y_sym['GS']) + sympy_params['True_N_SE'] * y_delay_sym['F_SRE']('F_SRE', 'sigma')
                  - sympy_params['beta'] * y_sym['S'] * (Infected / sympy_params['N']) - (sympy_params['ili'] + sympy_params['g_beta'] + sympy_params['phi_a_1'] + sympy_params['phi_a_2']) * y_sym['S'] )
# Exposed: E
f[states['E']] = (sympy_params['beta'] * y_sym['S'] + sympy_params['beta_prime'] * y_sym['FPI']) * (Infected / sympy_params['N']) - (sympy_params['per_a'] + sympy_params['per_s']) * y_sym['E']

# -------------- Asymptomatic Process -------------------
# Fictitious State before PA: F_PA
f[states['F_PA']] = sympy_params['per_a'] * y_sym['E'] - y_sym['F_PA']

# Pre Asymptomatic: PA
f[states['PA']] = y_sym['F_PA'] - y_delay_sym['F_PA']('F_PA', 'eta')

# Infected Asymptomatic : IA
f[states['IA']] = y_delay_sym['F_PA']('F_PA', 'eta') + y_sym['ATN'] - sympy_params['lambda_a'] * y_sym['IA'] - (sympy_params['phi_a_1'] + sympy_params['phi_a_2']) * y_sym['IA']

# Fictitious state before AT_1: F_AT_1
f[states['F_AT_1']] = sympy_params['phi_a_1'] * y_sym['IA'] - y_sym['F_AT_1']

# Asymptomatic Test 1 : AT_1
f[states['AT_1']] = y_sym['F_AT_1'] - y_delay_sym['F_AT_1']('F_AT_1', 'tau_1')

# Fictitious state before AT_2: F_AT_2
f[states['F_AT_2']] = sympy_params['phi_a_2'] * y_sym['IA'] - y_sym['F_AT_2']

# Asymptomatic Test 2 : AT_2
f[states['AT_2']] = y_sym['F_AT_2'] - y_delay_sym['F_AT_2']('F_AT_2', 'tau_2')

# Fictitious state before QAP: F_QAP
f[states['F_QAP']] = sympy_params['True_P_1'] * y_delay_sym['F_AT_1']('F_AT_1', 'tau_1') + sympy_params['True_P_2'] * y_delay_sym['F_AT_2']('F_AT_2', 'tau_2') - y_sym['F_QAP']

# Quarantined Asymp. Positive: QAP
f[states['QAP']] = y_sym['F_QAP'] - y_delay_sym['F_QAP']('F_QAP', 'lambda_q')

# Asymptomatic Tested Negative: ATN
f[states['ATN']] = (1 - sympy_params['True_P_1']) * y_delay_sym['F_AT_1']('F_AT_1', 'tau_1') + (1 - sympy_params['True_P_2']) * y_delay_sym['F_AT_2']('F_AT_2', 'tau_2') - y_sym['ATN']

# ------------- Systematic Testing Process --------------
# Fictitious state before PS: F_PS
f[states['F_PS']] = sympy_params['per_s'] * y_sym['E'] - y_sym['F_PS']

# Pre Symptomatic : PS
f[states['PS']] = y_sym['F_PS'] - y_delay_sym['F_PS']('F_PS', 'omega')

# Infected Symptomatic : IS
f[states['IS']] = y_delay_sym['F_PS']('F_PS', 'omega') + y_sym['STN'] - (sympy_params['phi_s_1'] + sympy_params['phi_s_2'] + sympy_params['lambda_s'] + sympy_params['kappa_s_1'] + sympy_params['hos_2']) * y_sym['IS']

# Fictitious state before ST_1: F_ST_1
f[states['F_ST_1']] = sympy_params['phi_s_1'] * y_sym['IS'] - y_sym['F_ST_1']

# Symptomatic Test 1 : ST_1
f[states['ST_1']] = y_sym['F_ST_1'] - y_delay_sym['F_ST_1']('F_ST_1', 'tau_1')

# Fictitious state before ST_2:  F_ST_2
f[states['F_ST_2']] = sympy_params['phi_s_2'] * y_sym['IS'] - y_sym['F_ST_2']

# Symptomatic Test 2 : ST_2
f[states['ST_2']] = y_sym['F_ST_2'] - y_delay_sym['F_ST_2']('F_ST_2', 'tau_2')

# Fictitious state before QSP: F_QSP
f[states['F_QSP']] = sympy_params['True_P_1']*y_delay_sym['F_ST_1']('F_ST_1', 'tau_1') + sympy_params['True_P_2']*y_delay_sym['F_ST_2']('F_ST_2', 'tau_2') - y_sym['F_QSP']

# Quarantined Symptomatic Positive : QSP
f[states['QSP']] = (1 - (sympy_params['hos_1'] + sympy_params['kappa_s_2'])) * ( y_sym['F_QSP'] - y_delay_sym['F_QSP']('F_QSP', 'lambda_q'))

# Symptomatic Test Negative : STN
f[states['STN']] = (1 - sympy_params['True_P_1']) * y_delay_sym['F_ST_1']('F_ST_1', 'tau_1') + (1 - sympy_params['True_P_2']) * y_delay_sym['F_ST_2']('F_ST_2', 'tau_2') - y_sym['STN']

# Fictitious state before H1:  F_H1
f[states['F_H1']] = sympy_params['hos_1'] * y_sym['F_QSP'] + (1 - sympy_params['True_N_1']) * y_delay_sym['F_ST_4']('F_ST_4', 'tau_1') - y_sym['F_H1']

# Hospitalized : H1
f[states['H_1']] = (1 - sympy_params['kappa_h_1']) * (y_sym['F_H1'] - y_delay_sym['F_H1']('F_H1', 'lambda_H_1'))

# F_ST_4 : Fictitious state before ST_4
f[states['F_ST_4']] = (1 - sympy_params['kappa_h_1']) * y_delay_sym['F_H1']('F_H1', 'lambda_H_1') - y_sym['F_ST_4']

# Symptomatic Test 4 : ST_4
f[states['ST_4']] = y_sym['F_ST_4'] - y_delay_sym['F_ST_4']('F_ST_4', 'tau_1')

# Fictitious state before H2: F_H2
f[states['F_H2']] = sympy_params['hos_2'] * y_sym['IS'] + (1 - sympy_params['True_N_1']) * y_delay_sym['F_ST_3']('F_ST_3', 'tau_1') - y_sym['F_H2']

# Hospitalzied 2 : H2
f[states['H_2']] = (1 - sympy_params['kappa_h_2']) * ( y_sym['F_H2'] - y_delay_sym['F_H2']('F_H2', 'lambda_H_2'))

# Fictitious state before ST_3:  F_ST_3
f[states['F_ST_3']] = (1 - sympy_params['kappa_h_2']) * y_delay_sym['F_H2']('F_H2', 'lambda_H_2') - y_sym['F_ST_3']

# Symptomatic Test 3 : ST_3
f[states['ST_3']] = y_sym['F_ST_3'] - y_delay_sym['F_ST_3']('F_ST_3', 'tau_1')

# Known Recover : KR
f[states['KR']] = y_delay_sym['F_QAP']('F_QAP', 'lambda_q') + (1 - (sympy_params['hos_1'] + sympy_params['kappa_s_2'])) * y_delay_sym['F_QSP']('F_QSP', 'lambda_q') + y_delay_sym['F_QAP_1']('F_QAP_1', 'lambda_q') + sympy_params['True_N_1'] * (y_delay_sym['F_ST_3']('F_ST_3', 'tau_1') + y_delay_sym['F_ST_4']('F_ST_4', 'tau_1')) - y_sym['KR']

# Dead : D
f[states['D']] = sympy_params['kappa_s_1'] * y_sym['IS'] + sympy_params['kappa_s_2'] * y_sym['F_QSP'] + sympy_params['kappa_h_1'] * y_sym['F_H1'] + sympy_params['kappa_h_2'] * y_sym['F_H2']

# -------------- Flu Process --------------
# Susceptible with Flu : SF
f[states['SF']] = sympy_params['ili'] * y_sym['S'] - sympy_params['phi_s_1'] * y_sym['SF'] - sympy_params['phi_s_2'] * y_sym['SF'] - (1 - sympy_params['phi_s_1'] - sympy_params['phi_s_2']) * y_sym['SF']

# Fictitious state before FT_1: F_FT_1
f[states['F_FT_1']] = sympy_params['phi_s_1'] * y_sym['SF'] - y_sym['F_FT_1']

# Flu like symptoms Test 1 : FT_1
f[states['FT_1']] = y_sym['F_FT_1'] - y_delay_sym['F_FT_1']('F_FT_1', 'tau_1')

# Fictitious state before FT_2:  F_FT_2
f[states['F_FT_2']] = sympy_params['phi_s_2'] * y_sym['SF'] - y_sym['F_FT_2']

# Flu like symptoms Test 2 : FT_2
f[states['FT_2']] = y_sym['F_FT_2'] - y_delay_sym['F_FT_2']('F_FT_2', 'tau_2')

# Fictitious state before QFS:  F_QFS
f[states['F_QFS']] = (1 - sympy_params['True_N_1']) * y_delay_sym['F_FT_1']('F_FT_1', 'tau_1') + (1 - sympy_params['True_N_2']) * y_delay_sym['F_FT_2']('F_FT_2', 'tau_2') - y_sym['F_QFS']

# Quarantined flu like symptoms : QFS
f[states['QFS']] = (1 - sympy_params['hos_3']) * (y_sym['F_QFS'] - y_delay_sym['F_QFS']('F_QFS', 'lambda_q'))

# Flu like Symptoms Test Negative : FTN
f[states['FTN']] = sympy_params['True_N_1'] * y_delay_sym['F_FT_1']('F_FT_1', 'tau_1') + sympy_params['True_N_2'] * y_delay_sym['F_FT_2']('F_FT_2', 'tau_2') - y_sym['FTN']

# Fictitious state before H3:  F_H3
f[states['F_H3']] = sympy_params['hos_3'] * y_sym['F_QFS'] - y_sym['F_H3'] + (1 - sympy_params['True_N_1']) * y_delay_sym['F_FT_3']('F_FT_3', 'tau_1')

# Hospitalized 3 : H3
f[states['H_3']] = y_sym['F_H3'] - y_delay_sym['F_H3']('F_H3', 'lambda_H_3')

# Fictitious state before FT_3: F_FT_3
f[states['F_FT_3']] = y_delay_sym['F_H3']('F_H3', 'lambda_H_3') - y_sym['F_FT_3']

# Flu like symptoms Test 3 : FT_3
f[states['FT_3']] = y_sym['F_FT_3'] - y_delay_sym['F_FT_3']('F_FT_3', 'tau_1')

# ------------ Non_Infected Process ---------------
# Fictitious state before NT_1: F_NT_1
f[states['F_NT_1']] = sympy_params['phi_a_1'] * y_sym['S'] - y_sym['F_NT_1']

# Non infected Test 1 :  NT_1
f[states['NT_1']] = y_sym['F_NT_1'] - y_delay_sym['F_NT_1']('F_NT_1', 'tau_1')

# Fictitious state before NT_2:  F_NT_2
f[states['F_NT_2']] = sympy_params['phi_a_2'] * y_sym['S'] - y_sym['F_NT_2']

# Non infected Test 2 :  NT_2
f[states['NT_2']] = y_sym['F_NT_2'] - y_delay_sym['F_NT_2']('F_NT_2', 'tau_2')

# Fictitious state before NTP: F_NTP
f[states['F_NTP']] = (1 - sympy_params['True_N_1']) * y_delay_sym['F_NT_1']('F_NT_1', 'tau_1') + (1 - sympy_params['True_N_2']) * y_delay_sym['F_NT_2']('F_NT_2', 'tau_2') - y_sym['F_NTP']

# Quarantined Non infected test positvie : NTP
f[states['NTP']] = y_sym['F_NTP'] - y_delay_sym['F_NTP']('F_NTP', 'lambda_q')

# Non infected Test Negative : NTN
f[states['NTN']] = sympy_params['True_N_1'] * y_delay_sym['F_NT_1']('F_NT_1', 'tau_1') + sympy_params['True_N_2'] * y_delay_sym['F_NT_2']('F_NT_2', 'tau_2') - y_sym['NTN']

# General Sick Process: GS
f[states['GS']] = sympy_params['g_beta'] * y_sym['S'] - sympy_params['phi_s_1'] * y_sym['GS'] - sympy_params['phi_s_2'] * y_sym['GS'] - (1 - sympy_params['phi_s_1'] - sympy_params['phi_s_2']) * y_sym['GS']

# Fictitious before GT_1: F_GT1
f[states['F_GT1']] = sympy_params['phi_s_1'] * y_sym['GS'] - y_sym['F_GT1']

# General Sick Test:  GT_1
f[states['GT_1']] = y_sym['F_GT1'] - y_delay_sym['F_GT1']('F_GT1', 'tau_1')

# Fictitious before GT_2: F_GT_2
f[states['F_GT2']] = sympy_params['phi_s_2'] * y_sym['GS'] - y_sym['F_GT2']

# General Sick Test:  GT_2
f[states['GT_2']] = y_sym['F_GT2'] - y_delay_sym['F_GT2']('F_GT2', 'tau_2')

# Fictitious before QGP: F_QGP
f[states['F_QGP']] = (1 - sympy_params['True_N_1']) * y_delay_sym['F_GT1']('F_GT1', 'tau_1') + (1 - sympy_params['True_N_2']) * y_delay_sym['F_GT2']('F_GT2', 'tau_2') - y_sym['F_QGP']

# General Sick Test Negative: GTN
f[states['GTN']] = sympy_params['True_N_1'] * y_delay_sym['F_GT1']('F_GT1', 'tau_1') + sympy_params['True_N_2'] * y_delay_sym['F_GT2']('F_GT2', 'tau_2') - y_sym['GTN']

# Quarantined General Sick: QGP
f[states['QGP']] = y_sym['F_QGP'] - y_delay_sym['F_QGP']('F_QGP', 'lambda_q')

# F_FPS : Fictitious state before FPS (Xin Code)
f[states['F_FPS']] = y_sym['UR'] + y_sym['ATN_1'] + (1 - sympy_params['True_P_SE']) * y_delay_sym['F_STI']('F_STI', 'sigma') - y_sym['F_FPS']

# Falsely Presumed Susceptible : FPS
f[states['FPS']] = (1 - (sympy_params['phi_a_1'] + sympy_params['phi_a_2'])) * (y_sym['F_FPS'] - y_delay_sym['F_FPS']('F_FPS', 'gamma'))

# Fictitious state AT_3: F_AT_3
f[states['F_AT_3']] = sympy_params['phi_a_1'] * y_sym['F_FPS'] - y_sym['F_AT_3']

# Asymptomatic Test 3 : AT_3
f[states['AT_3']] = y_sym['F_AT_3'] - y_delay_sym['F_AT_3']('F_AT_3', 'tau_1')

# Fictitious state AT_4: F_AT_4
f[states['F_AT_4']] = sympy_params['phi_a_2'] * y_sym['F_FPS'] - y_sym['F_AT_4']

# Asymptomatic Test 4 : AT_4
f[states['AT_4']] = y_sym['F_AT_4'] - y_delay_sym['F_AT_4']('F_AT_4', 'tau_2')

# Fictitious state QAP_1: F_QAP_1
f[states['F_QAP_1']] = (1 - sympy_params['True_N_1']) * y_delay_sym['F_AT_3']('F_AT_3', 'tau_1') + (1 - sympy_params['True_N_2']) * y_delay_sym['F_AT_4']('F_AT_4', 'tau_2') - y_sym['F_QAP_1']

# Quarantined Asymptomatic Positve_1 : QAP_1
f[states['QAP_1']] = y_sym['F_QAP_1'] - y_delay_sym['F_QAP_1']('F_QAP_1', 'lambda_q')

# Asymptomatic Tested Negative_1 : ATN_1
f[states['ATN_1']] = sympy_params['True_N_1'] * y_delay_sym['F_AT_3']('F_AT_3', 'tau_1') + sympy_params['True_N_2'] * y_delay_sym['F_AT_4']('F_AT_4', 'tau_2') - y_sym['ATN_1']

# ----------- Immunity Process -------------
# Fictitious state before IM: F_IM
f[states['F_IM']] = y_sym['KR'] - y_sym['F_IM']

# Immunity : IM
f[states['IM']] = (1 - sympy_params['phi_se']) * ( y_sym['F_IM'] - y_delay_sym['F_IM']('F_IM', 'gamma')) + sympy_params['True_P_SE'] * y_delay_sym['F_STI']('F_STI', 'sigma')

# Fictitious state before STI: F_STI
f[states['F_STI']] = sympy_params['phi_se'] * y_sym['F_IM'] - y_sym['F_STI']

# Serology Test for Immuned : STI (Xin Code)
f[states['STI']] = y_sym['F_STI'] - y_delay_sym['F_STI']('F_STI', 'sigma')

# Falsely Presumed Immune : FPI (Xin Code)
f[states['FPI']] = (y_delay_sym['F_QGP']('F_QGP', 'lambda_q') + y_delay_sym['F_NTP']('F_NTP', 'lambda_q') + (1 - sympy_params['hos_3']) * y_delay_sym['F_QFS']('F_QFS', 'lambda_q')
                   + (1 - sympy_params['phi_se']) * y_delay_sym['F_IM']('F_IM', 'gamma') + (1 - sympy_params['True_N_SE']) * y_delay_sym['F_SRE']('F_SRE', 'sigma')
                   + sympy_params['True_N_1'] * y_delay_sym['F_FT_3']('F_FT_3', 'tau_1') - sympy_params['phi_se'] * y_sym['FPI'] - sympy_params['beta_prime'] * y_sym['FPI'] * Infected / sympy_params['N'])

# Fictitious state before SRE: F_SRE
f[states['F_SRE']] = sympy_params['phi_se'] * y_sym['FPI'] - y_sym['F_SRE']

# Serology Test for Immune Expired: SRE
f[states['SRE']] = y_sym['F_SRE'] - y_delay_sym['F_SRE']('F_SRE', 'sigma')

# Unknown Recover : UR
f[states['UR']] = sympy_params['lambda_a'] * y_sym['IA'] + sympy_params['lambda_s'] * y_sym['IS'] - y_sym['UR']
