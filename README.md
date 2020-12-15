# building-characteristics
Repository for code used to analyze environmental CO2 and temperature data, so that heat loss characteristics of a room can be calculated. Analyses made to check results against base estimations

The code in /U-value-calculation analyses raw environmental (CO2 and temperature) data to calculate the U-value, or thermal efficiency of a room's solid elements. This is done by fitting exponential decay curves to both elements, and estimating their decay coefficients. Decay coefficients will only be kept for calculations when decays overlap/happen at the same time.
A Monte Carlo analysis is used to minimise the propagation of errors as the equation to find the U-value is applied using these coefficients.
The results which have been found can be averaged to find an accurate U-value.

/calcs-with-graphs contains the code with associated graphics.

/analysis-of-variables contains an analysis of the effect of different variables in the U-value equation on the overall end result.
