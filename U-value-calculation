import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datetime import datetime
import matplotlib.dates as mdates
import math
import scipy.stats as stats

# Open + read co2 and temperature data files
with open('BH_co2_data.csv', 'r') as co2_file:
    csv_reader = csv.reader(co2_file, delimiter=',')
    # Empty lists store gas concentration values, time and measurement number
    time_day = []
    gas_conc = []
    measure_num = []    # counts measurement number in the file
    count = 0
    for row in csv_reader:
        time_day.append(datetime.strptime(row[1], "%d/%m/%Y %H:%M"))     # make it datetime format
        gas_conc.append(float(row[2]))
        measure_num.append(count)
        count = count + 1
# Name x (time) and y (gas concentration) data
x_array = np.array(measure_num)
y_array = np.array(gas_conc)

with open('BH_temp_data.csv', 'r') as temp_file:
    csv_reader = csv.reader(temp_file, delimiter=',')
    time_day2 = []
    temperature = []
    measure_num2 = []
    count = 0
    for row in csv_reader:
        time_day2.append(datetime.strptime(row[1], "%d/%m/%Y %H:%M"))
        temperature.append(float(row[2]))
        measure_num2.append(count)
        count = count + 1
# Name t (time) and h (temperature) data
t_array = np.array(measure_num2)
h_array = np.array(temperature)

decay_period = []
decayID = []
U_val_list = []
U_val_errors = []


# define an exponential function to be fitted to the data
def exponential(X, A, K, C):
    return A * np.exp(-K * X) + C


# intersection check function
def date_intersection(tA, tB):
    tAstart, tAend = tA[1], tA[2]
    tBstart, tBend = tB[1], tB[2]
    if tAstart <= tBstart <= tAend:
        latest_start = max(tBstart, tAstart)    # find overlap start
        earliest_end = min(tBend, tAend)    # find overlap end
        decay_period.append(latest_start)
        decay_period.append(earliest_end)
    elif tAstart <= tBend <= tAend:
        latest_start = max(tBstart, tAstart)
        earliest_end = min(tBend, tAend)
        decay_period.append(latest_start)
        decay_period.append(earliest_end)
    elif tBstart < tAstart and tAend < tBend:
        latest_start = max(tBstart, tAstart)
        earliest_end = min(tBend, tAend)
        decay_period.append(latest_start)
        decay_period.append(earliest_end)
    else:
        decay_period.clear()
    return decay_period     # prints start and finish times

# For the variation of U
def randomize_U():
    """normal(loc, scale, size) generates a normal continuous random variable.
        loc=mean, scale=standard deviation, size=number_of_generations
       uniform(min, max)"""
    y_value = normal(0.15, 0.15*0.1)      # thermal bridging, error of 10% due to approximative nature of variable
    kay = normal(kappa, kappa_err)
    elle = normal(lamda, lamda_err)
    Cv_air = 0.7172 * 1000      # constant
    length = uniform((12.7 - 0.1), (12.7 + 0.1))     # length +/- 0.1 m
    height = uniform((2 - 0.1), (2 + 0.1))
    width = uniform((9 - 0.1), (9 + 0.1))
    thickness = uniform((0.5 - 0.01), (0.5 + 0.01))  # thickness +/- 0.01 m
    A_win = 5.616  # window area, constant
    den_wall = normal(2611, (2611 * 0.05))  # densities with 5% error
    den_win = normal(2579, (2579 * 0.05))  # density of the window as it's an external element
    den_a = 1.225  # constant
    Cp_wall = normal(0.91, (0.91 * 0.05))  # specific heat capacities with 5% error
    Cp_win = normal(0.84, (0.84 * 0.05))
    Cp_a = 1.006  # constant
    Uvalue = kay * ((Cp_a * den_a * (width * length * height) + Cp_wall * den_wall * (length * height * thickness) + \
         Cp_win * den_win * A_win * thickness)*1000 - Cv_air * width * length * height * elle) * \
             (1 / (length * height)) - y_value
    return Uvalue


# Finding decay and decrement in data
amt_decrease = 0
increment = []
decay_counter = 0
gas_decay_list = []
for i in measure_num:
    if gas_conc[i] <= gas_conc[i - 1]:
        amt_decrease += 1
        increment.append(time_day[i])
    elif gas_conc[i] <= gas_conc[i - 2] and (gas_conc[i] - gas_conc[i - 1]) < 40:
        amt_decrease += 1
        increment.append(time_day[i])
    elif gas_conc[i] <= gas_conc[i - 3] and (gas_conc[i] - gas_conc[i - 1]) < 20:
        amt_decrease += 1
        increment.append(time_day[i])
    elif gas_conc[i] <= gas_conc[i - 4] and (gas_conc[i] - gas_conc[i - 1]) < 5:
        amt_decrease += 1
        increment.append(time_day[i])
        continue
    else:
        if amt_decrease > 10:
            decay_counter += 1
            # Find decay coefficient (l) for gas concentration decay
            sliced_x = np.arange(start=0, stop=(amt_decrease*3600), step=3600)       # turn hour step into seconds
            sliced_y = np.array(gas_conc[i - amt_decrease:i])
            # Estimate start values
            a0 = (sliced_y[0] - sliced_y[-1])
            l0 = 1/3600
            c0 = sliced_y[-1]
            start = a0, l0, c0
            # Fit the curve to the data
            bestfit, covar = curve_fit(exponential, sliced_x, sliced_y, p0=start,
                                       maxfev=100000)
            a, l, c = bestfit
            coverr = np.sqrt(np.diag(covar))
            l_error = coverr[1]
            print(l, l_error)

            # Save data for when decay periods overlap so it can be plotted
            st = increment[0]
            ed = increment[-1]
            gas_decay_list.append([decay_counter, st, ed, l, increment, gas_conc[i - amt_decrease:i], l_error])  # chunk
        amt_decrease = 0
        increment = []
        continue

# Decrement in temperature data
amt_decrease2 = 0
increment2 = []
decay_counter2 = 0
temp_decay_list = []
for i in measure_num2:
    if temperature[i] <= temperature[i - 1]:
        amt_decrease2 += 1
        increment2.append(time_day2[i])
    elif temperature[i] <= temperature[i - 2] and (temperature[i] - temperature[i - 1]) < 0.5:
        amt_decrease2 += 1
        increment2.append(time_day2[i])
    elif temperature[i] <= temperature[i - 3] and (temperature[i] - temperature[i - 1]) < 0.1:
        amt_decrease2 += 1
        increment2.append(time_day2[i])
    elif temperature[i] <= temperature[i - 4] and (temperature[i] - temperature[i - 1]) < 0.05:
        amt_decrease2 += 1
        increment2.append(time_day2[i])
        continue
    else:
        if amt_decrease2 > 8:
            decay_counter2 += 1
            sliced_t = np.arange(start=0, stop=(amt_decrease2*3600), step=3600)
            sliced_h = np.array(temperature[i - amt_decrease2:i])
            # Start parameters
            b0 = (sliced_h[0] - sliced_h[-1])
            k0 = 1 / 3600
            d0 = sliced_h[-1]
            start = b0, k0, d0
            # Fit curve
            bestfit, covar = curve_fit(exponential, sliced_t, sliced_h, p0=start,
                                       maxfev=100000)
            b, k, d = bestfit
            coverr = np.sqrt(np.diag(covar))
            k_error = coverr[1]
            print(k, k_error)

            # Save data
            st = increment2[0]
            ed = increment2[-1]
            temp_decay_list.append([decay_counter2, st, ed, k, increment2, temperature[i - amt_decrease2:i], k_error])
        amt_decrease2 = 0
        increment2 = []
        continue

# Checking for intersections in the decays for each decay period
for i in gas_decay_list:
    tA = i
    gdecayID = i[0]     # decay "IDs" make data easier to check visually
    for j in temp_decay_list:
        tB = j
        hdecayID = j[0]
        intersection = date_intersection(tA, tB)
        if intersection:        # check if list has elements in it
            # Retrieve decay coefficients for respective decays
            for m in gas_decay_list:
                if m[0] == gdecayID:
                    lamda = m[3]
                    lamda_err = m[6]
            for n in temp_decay_list:
                if n[0] == hdecayID:
                    kappa = n[3]
                    kappa_err = n[6]
  
            # Calculate U-value
            # Monte Carlo analysis of variables allows a standard deviation of these values to be calculated
            N = 1000
            sim = np.zeros(N)
            for i in range(N):
                sim[i] = randomize_U()
            # Get rid of results with large errors/standard deviations
            if np.std(sim) < 5:
                U_err = np.std(sim)
                U_avg = np.average(sim)
                # Update list for averaging once runthrough is completed
                U_val_list.append(U_avg) 
                U_val_errors.append(U_err)

        continue        # go back and iterate to check for any other overlap in the intersection
    continue        # continue looping for all intersections

# Average calculated U-values
U_tot_avg = sum(U_val_list) / len(U_val_list)
U_tot_err = sum(U_val_errors) / len(U_val_errors)
print(U_tot_avg, U_tot_err)
