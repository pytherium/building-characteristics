import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datetime import datetime
import matplotlib.dates as mdates
import math
import scipy.stats as stats
from numpy.random import *
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator

# Inputs
avg_u = 2.67        # average U-value of the room, estimated manually using only exposed fabric (W/m2K)

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
        time_day2.append(datetime.strptime(row[1], "%d/%m/%Y %H:%M"))       # make it datetime format?
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
            # PLOT CO2
            color = 'tab:blue'
            plt.fig1 = plt.figure(1, figsize=(7, 5.5))
            y1 = exponential(sliced_x, a + coverr[0], l + coverr[1], c + coverr[2])
            y2 = exponential(sliced_x, a - coverr[0], l - coverr[1], c - coverr[2])
            # Plot data
            plt.f1 = plt.fig1.add_axes((.1, .1, .8, .6))
            plt.plot(increment, sliced_y, marker='.', color=color, label="Measured CO2 data")
            plt.plot(increment, exponential(sliced_x, *bestfit), '-', color='grey', lw=1,
                     label=f'y= {a:.2f}e^{l:.2f}t+{c:.2f}')
            plt.plot(increment, y1, c='lightgray', linewidth=0.1)
            plt.plot(increment, y2, c='lightgray', linewidth=0.1)
            plt.fill_between(increment, y1, y2, facecolor="lightgray", label="Fit error of curve")
            plt.ylabel('CO2 Concentration (ppm)', color=color)
            plt.xlabel('DateTime')
            plt.tick_params(axis='y', colors=color)
            plt.legend(loc='upper right')
            # Residuals plot
            plt.f2 = plt.fig1.add_axes((.1, .7, .8, .2))
            plt.plot(sliced_x, sliced_y - exponential(sliced_x, *bestfit), '.', color=color)
            plt.hlines(0, 0, max(sliced_x), color='grey', lw=1)
            plt.f2.set_xticklabels([])  # remove label on x axis
            plt.xticks([])  # remove ticks on x axis
            plt.f1.xaxis.set_minor_locator(mdates.HourLocator(byhour=(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22)))
            plt.f1.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))
            plt.f1.xaxis.set_major_locator(mdates.DayLocator())
            plt.f1.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
            plt.f1.xaxis.remove_overlapping_locs = False
            plt.f1.tick_params(axis="x", which="major", pad=10)
            plt.f1.tick_params(axis="x", which="minor", labelsize=8)
            plt.ylabel('Fit residuals', color=color)
            plt.tick_params(axis='y', labelcolor=color)
            plt.show()

            # Save data for when decay periods overlap so it can be plotted
            st = increment[0]
            ed = increment[-1]
            gas_decay_list.append([decay_counter, st, ed, l, increment, gas_conc[i - amt_decrease:i], l_error])
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
            # PLOT TEMP
            color = 'tab:red'
            plt.fig1 = plt.figure(1, figsize=(7, 5.5))
            y1 = exponential(sliced_t, b + coverr[0], k + coverr[1], d + coverr[2])
            y2 = exponential(sliced_t, b - coverr[0], k - coverr[1], d - coverr[2])
            # Plot data
            plt.f1 = plt.fig1.add_axes((.1, .1, .8, .6))
            plt.plot(increment2, sliced_h, marker='.', color=color, label="Measured temperature data")
            plt.plot(increment2, exponential(sliced_t, *bestfit), '-', color='grey', lw=1,
                     label=f'y= {b:.2f}e^{k:.2f}t+{d:.2f}')
            plt.plot(increment2, y1, c='lightgray', linewidth=0.1)
            plt.plot(increment2, y2, c='lightgray', linewidth=0.1)
            plt.fill_between(increment2, y1, y2, facecolor="lightgray", label="Fit error of curve")
            plt.ylabel('CO2 Concentration (ppm)', color=color)
            plt.xlabel('DateTime')
            plt.tick_params(axis='y', colors=color)
            plt.legend(loc='upper right')
            # Residuals plot
            plt.f2 = plt.fig1.add_axes((.1, .7, .8, .2))
            plt.plot(sliced_t, sliced_h - exponential(sliced_t, *bestfit), '.', color=color)
            plt.hlines(0, 0, max(sliced_t), color='grey', lw=1)
            plt.f2.set_xticklabels([])  # remove label on x axis
            plt.xticks([])  # remove ticks on x axis
            plt.f1.xaxis.set_minor_locator(mdates.HourLocator(byhour=(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22)))
            plt.f1.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))
            plt.f1.xaxis.set_major_locator(mdates.DayLocator())
            plt.f1.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
            plt.f1.xaxis.remove_overlapping_locs = False
            plt.f1.tick_params(axis="x", which="major", pad=10)
            plt.f1.tick_params(axis="x", which="minor", labelsize=8)
            plt.ylabel('Fit residuals', color=color)
            plt.tick_params(axis='y', labelcolor=color)
            plt.show()

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
            ## Calculate U-value
            # Monte Carlo analysis of variables
            N = 1000
            sim = np.zeros(N)
            for i in range(N):
                sim[i] = randomize_U()

            if np.std(sim) < 5:
                U_err = np.std(sim)
                U_avg = np.average(sim)
                U_val_list.append(U_avg) 
                U_val_errors.append(U_err)
                
            fig, ax1 = plt.subplots()
            ax1.hist(sim, color="lightgray", label="Simulation output")
            plt.vlines(U_avg, 0, 270, color="#900C8D", linestyles='dashed', label=f"Mean: {U_avg:.2f}")
            plt.vlines(U_avg + U_err, 0, 200, color="#900C8D", linestyles=(0, (1, 1)),
                       label=f"St. deviation: +/- {U_err:.2f}")
            plt.vlines(U_avg - U_err, 0, 200, color="#900C8D", linestyles=(0, (1, 1)))
            plt.xlabel("U-value, W/m²K")
            plt.ylabel("Frequency")
            plt.legend(loc='upper left', fontsize='small')
            ax2 = ax1.twinx()
            domain = np.linspace(min(sim), max(sim), 500)
            ax2.plot(domain, stats.norm.pdf(domain, U_avg, U_err), color="grey", label="PDF")
            plt.ylabel("Probability")
            plt.legend(loc='upper right', fontsize='medium')
            plt.tight_layout()
            plt.show()

            ## Plot the intersection of both decays
            # Retrieve data directly from csv file
            with open('BH_co2_data.csv', 'r') as co2_file:
                csv_reader = csv.reader(co2_file, delimiter=',')
                for row in csv_reader:
                    if row[1] == datetime.strftime(intersection[0], "%d/%m/%Y %H:%M"):       # if the time corresponds to the time at the start of intersecn
                        st_conc = int(row[0])       # the start concentration is equal to row number
                    if row[1] == datetime.strftime(intersection[1], "%d/%m/%Y %H:%M"):
                        ed_conc = int(row[0])
            with open('BH_temp_data.csv', 'r') as temp_file:
                csv_reader = csv.reader(temp_file, delimiter=',')
                for row in csv_reader:
                    if row[1] == datetime.strftime(intersection[0], "%d/%m/%Y %H:%M"):
                        st_temp = int(row[0])         # the start temp is equal to row number
                    if row[1] == datetime.strftime(intersection[1], "%d/%m/%Y %H:%M"):
                        ed_temp = int(row[0])
            # Plot intersection of decays
            fig, ax1 = plt.subplots()
            color = 'tab:blue'
            ax1.set_xlabel('DateTime', fontweight='bold')
            ax1.set_ylabel('CO2  Concentration (ppm)',  fontweight='bold', color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.plot(time_day[st_conc:ed_conc], gas_conc[st_conc:ed_conc], color=color)

            ax2 = ax1.twinx()
            color = 'tab:red'
            ax2.set_ylabel('Temperature (°C)',  fontweight='bold', color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.plot(time_day2[st_temp:ed_temp], temperature[st_temp:ed_temp], color=color)

            # format ticks
            ax1.xaxis.set_minor_locator(mdates.HourLocator(byhour=(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22)))
            ax1.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))
            ax1.xaxis.set_major_locator(mdates.DayLocator())
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
            ax1.xaxis.remove_overlapping_locs = False
            ax1.tick_params(axis="x", which="major", pad=10)
            ax1.tick_params(axis="x", which="minor", labelsize=8)
            fig.tight_layout()
            plt.show()
        continue        # go back and iterate to check for any other overlap in the intersection
    continue        # continue looping for all intersections
U_tot_avg = sum(U_val_list) / len(U_val_list)
U_tot_err = sum(U_val_errors) / len(U_val_errors)
print(U_tot_avg, U_tot_err)

# Find error of individual U-value calculations
measures = np.arange(start=1, stop=len(U_val_list)+1, step=1)
values =  np.array(U_val_list)
errors = np.array(U_val_errors)
# Plot figure of U-values as compared to an estimated U-value
fig, ax = plt.subplots(figsize=(7, 4))
est_err = avg_u * 2        # 200 % error from estimate on plot
plt.hlines(avg_u, 1, len(U_val_list), 'grey', label="Estimated room U-value")
plt.fill_between(measures, 0, avg_u + est_err, facecolor='lightgray', label="200% error margin")
plt.errorbar(measures, values, yerr=U_val_errors, fmt='.', color='#900C8D', elinewidth=1, capsize=1.5,
             label="Calculated U-values")
plt.xlabel("Estimation of U-value per intersecting decays")
plt.ylabel("U-value, W/m²K")
ax.xaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_minor_locator(MultipleLocator(1))
plt.legend(loc='best', fontsize='small')
plt.show()
