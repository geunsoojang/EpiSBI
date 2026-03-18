import random
import numpy as np
import csv
from DDE_s import *
from Parameters_s import *
from scipy.optimize import minimize



class InfectionModel:

    # reading the training data and initializing
    def __init__(self, train_dat_file, timespan):

        self.Beta = []
        self.Phi = []
        self.G_rate = []

        self.training_daily_pos = []
        self.training_daily_neg = []
        # read from the dat_file and store it
        with open(train_dat_file, 'r') as csvfile:
            reader = csv.reader(csvfile)

            # Loop through rows
            r = 0
            for row in reader:
                if r == 0:
                    r += 1
                    continue
                if r > timespan:
                    break
                positive_value = float(row[2])  
                negative_value = float(row[1])  

                # Append values to respective arrays
                self.training_daily_pos.append(positive_value)
                self.training_daily_neg.append(negative_value)

                r += 1

        self.duration = len(self.training_daily_pos)

    # Objective Function
    def opt_func(self, w):
        # params: w has [(a1, a2, b1, b2, g, eb, es), ...] of length (duration of the training data / 7)
        
        # reshape the parameters
        w = np.array(w).reshape(int(len(w) / 7), 7)

        inf_rate = [0.99, 0.84]
        phi_s = [0.09, 0.08]
        g_rate = [0.004, 0.004]
        e_b = [w[0][5], w[1][5]]
        e_s = [w[0][6], w[1][6]]

        # Calculate the inf rates, phi rates, g rates
        for k in range(2, len(w)):
            a_k, b_k, g_k, e_b_k, e_s_k = w[k][:2], w[k][2:4], w[k][4], w[k][5], w[k][6]
            beta_k = a_k[0] * inf_rate[k-1] + a_k[1] * inf_rate[k-2] + e_b_k
            phi_k = b_k[0] * phi_s[k-1] + b_k[1] * phi_s[k-2] + e_s_k

            inf_rate.append(beta_k)
            phi_s.append(phi_k)
            g_rate.append(g_k)
            e_b.append(e_b_k)
            e_s.append(e_s_k)
        
        # run the model with given parameters
        yy = runModel(init_cond, T_delay, params, self.duration, inf_rate, phi_s, g_rate)

        # =============== Calculate the daily pos and daily negative from the SIRTEM model sim ====================

        est_daily_pos = [ (yy[t][states['F_AT_1']] + yy[t][states['F_ST_1']])*(params['True_P_1']) +
                            (yy[t][states['F_AT_2']] + yy[t][states['F_ST_2']])*(params['True_P_2']) +
                            (1-params["True_N_1"]) * (yy[t][states['F_NT_1']] + yy[t][states['F_AT_3']] + 
                                                    yy[t][states['F_FT_1']] + yy[t][states['F_GT1']] + 
                                                    yy[t][states['F_ST_4']] + yy[t][states['F_ST_3']] + 
                                                    yy[t][states['F_FT_3']]) +
                            (1 - params["True_N_2"]) * (yy[t][states['F_NT_2']] + yy[t][states['F_AT_4']] + 
                                                    yy[t][states['F_FT_2']] + yy[t][states['F_GT2']] ) for t in range(self.duration)]

        
        est_daily_neg = [ (yy[t][states['F_AT_1']] + yy[t][states['F_ST_1']])*(1 - params['True_P_1']) +
                            (yy[t][states['F_AT_2']] + yy[t][states['F_ST_2']])*(1 - params['True_P_2']) +
                            params["True_N_1"] * (yy[t][states['F_NT_1']] + yy[t][states['F_AT_3']] + 
                                                    yy[t][states['F_FT_1']] + yy[t][states['F_GT1']] + 
                                                    yy[t][states['F_ST_4']] + yy[t][states['F_ST_3']] + 
                                                    yy[t][states['F_FT_3']]) +
                            params["True_N_2"] * (yy[t][states['F_NT_2']] + yy[t][states['F_AT_4']] + 
                                                    yy[t][states['F_FT_2']] + yy[t][states['F_GT2']] ) for t in range(self.duration)]
        # ========================= Objective Function ========================
        Z = (0.5 * (np.sum([(est_daily_pos[t] - self.training_daily_pos[t])**2 for t in range(self.duration)]) / (self.duration * np.mean(self.training_daily_pos)))
                    + 0.5 * (np.sum([(est_daily_neg[t] - self.training_daily_neg[t])**2 for t in range(self.duration)]) / (self.duration * np.mean(self.training_daily_neg))))
        
        penalty = 0
        for i in range(len(inf_rate)):
            if inf_rate[i] < 0:
                penalty += np.abs(inf_rate[i])
            elif inf_rate[i] > 1:
                penalty += (inf_rate[i] - 1)    
            if phi_s[i] < 0:
                penalty += np.abs(phi_s[i])
            elif phi_s[i] > 1:
                penalty += (phi_s[i] - 1)                
        
        print("=====>", Z, penalty)
        return Z + penalty

    # Calibration Function with initial parameters, bounds and constraints
    def calibration(self):

        # num of weeks
        K = int(self.duration / 7) + 1 
        params = np.array([(0.3, 0.3, 0.3, 0.3, 0.004, -0.05, -0.05) for k in range(K)]).flatten()
        
        # Define the bounds for (a1, a2, b1, b2, g, error_B and error_S)
        bnds = []
        for k in range(K):
            bnds += [(0, 2), (0, 2), (0, 2), (0, 2), (0, 0.005), (-0.5, 0.5), (-0.01, 0.01)]
        
        # Define the constraints that the infection rates and Phi rates must be between [0, 1]
        cons = []
        for k in range(2, K):
            def inf_cons(w, k=k):
                # reshape the parameters
                w = np.array(w).reshape(int(len(w) / 7), 7)
                # inf_rate = [w[0][0] * 0.5 + w[0][1] * 0.5 + w[0][5], w[1][0] * 0.5 + w[1][1] * 0.5 + w[1][5]]
                inf_rate = [0.99, 0.84]
                for k in range(2, len(w)):
                    a_k, b_k, g_k, e_b_k, e_s_k = w[k][:2], w[k][2:4], w[k][4], w[k][5], w[k][6]
                    beta_k = a_k[0] * inf_rate[k-1] + a_k[1] * inf_rate[k-2] + e_b_k

                    inf_rate.append(beta_k)
                return 1 - inf_rate[k]
            
            def phi_cons(w, k=k):
                # reshape the parameters
                w = np.array(w).reshape(int(len(w) / 7), 7)
                # phi_s = [w[0][2] * 0.5 + w[0][3] * 0.5 + w[0][6], w[1][2] * 0.5 + w[1][3] * 0.5 + w[1][6]]
                phi_s = [0.09, 0.08]
                for k in range(2, len(w)):
                    a_k, b_k, g_k, e_b_k, e_s_k = w[k][:2], w[k][2:4], w[k][4], w[k][5], w[k][6]
                    phi_k = b_k[0] * phi_s[k-1] + b_k[1] * phi_s[k-2] + e_s_k

                    phi_s.append(phi_k)
                return 1 - phi_s[k]

            cons.append({'type':'ineq', 'fun': inf_cons})
            cons.append({'type':'ineq', 'fun': phi_cons})


        # Run the Optimizer with the constraints and bounds
        result = minimize(self.opt_func, params, bounds=bnds, constraints=cons, method='SLSQP', options={'eps': 1.0})

        print(result.status, result.message)

        result_x = np.array(result.x).reshape(int(len(result.x) / 7), 7)

        pickle.dump(result_x, open("inf_rate_opt_unfmt.p", "wb"))
        result_x = pickle.load(open("inf_rate_opt_unfmt.p", "rb"))

        # ------------------------------------------------------------------------------
        # Use the 7 value tuples to calculate the Infection Rates, Phi Rates and G Rates
        inf_rate = [0.99, 0.84]
        phi_s = [0.09, 0.08]
        g_rate = [0.004, 0.004]

        for k in range(2, len(result_x)):
            a_k, b_k, g_k = result_x[k][:2], result_x[k][2:4], result_x[k][4]
            beta_k = a_k[0] * inf_rate[k-1] + a_k[1] * inf_rate[k-2] + result_x[k][5]
            phi_k = b_k[0] * phi_s[k-1] + b_k[1] * phi_s[k-2] + result_x[k][6]

            inf_rate.append(beta_k)
            phi_s.append(phi_k)
            g_rate.append(g_k)
        
        self.Beta = inf_rate
        self.Phi = phi_s
        self.G_rate = g_rate

        return self.Beta, self.Phi, self.G_rate

# Define the path to your training data file
train_dat_file = 'training_data_AZ.csv'

# Create an object of the InfectionModel class
timespan = 30
infection_model = InfectionModel(train_dat_file, timespan)

# Run the calibration method
calibration_result = tuple( infection_model.calibration() )

# Print the calibration result
print("Calibration result:", calibration_result)
pickle.dump(calibration_result, open("inf_rate_opt2.p", "wb"))
