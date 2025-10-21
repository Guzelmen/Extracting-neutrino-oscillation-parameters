"""
Plots for the first tasks, including data visualisation
"""

import numpy as np
import matplotlib.pyplot as plt


#Importing data from my computer. I took the online file and separated it
#into 2 .txt files for importation
data = np.array(np.loadtxt("experimental_data.txt", skiprows=1))
pred = np.array(np.loadtxt("unoscillated_event_rate_prediction.txt", skiprows=1))

x_range = np.linspace(0, 10, 200)#Bin edges
e_range = np.linspace(0.025, 9.975, 200)#Bin centers

#Event histogram
fig = plt.figure
plt.hist(x_range, bins = 200, weights = data)
plt.xlabel(r"$Energy [GeV]$")
plt.ylabel(r"Number of measured events")


def osc_prob(E, theta = (np.pi/4), mass_diff = (2.4), L = 295):
    p = 1 - (np.sin(2*theta)**2)*(np.sin(1.267*mass_diff*1e-3*L/E)**2)
    return p
prob = osc_prob(e_range)

#Probability variation with energy
fig2 = plt.figure()
plt.plot(e_range, prob, color = "red")
plt.xlabel(r"$Energy [GeV]$")
plt.ylabel(r"Probability of oscillation")


osc_pred = []

for i in range(200):
    osc_pred.append(prob[i]*pred[i])


#Comparing unoscillated vs oscillated event rate prediction
fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.hist(x_range, bins = 200, weights = pred)
ax1.set_ylabel(r"Unoscillated event prediction")
ax1.set_xlabel(r"$Energy [GeV]$")

ax2.hist(x_range, bins = 200, weights = osc_pred)
ax2.set_ylabel(r"Oscillated event prediction")
ax2.set_xlabel(r"$Energy [GeV]$")
ax2.yaxis.set_label_position("right")


#Seeing the effect of detector resolution on the new reconstruced spectrum
def gaussian(x, mean, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / sigma) ** 2)

def convolution(x, spectrum, bin_centers, sigma):
    convolved = 0
    for height, center in zip(spectrum, bin_centers):
        convolved += height*0.05 * gaussian(x, center, sigma)

    return convolved

def nll_sigma(bin_number, events, unosc_pred, sigma):
    suma = 0
    total_pred = []
    for i in range(bin_number):
        total_pred.append(osc_prob(e_range[i], 0.701, 2.398)*unosc_pred[i]*0.683*e_range[i])
    for i in range(bin_number):
        convol = convolution(x_range[i], total_pred, e_range, sigma) 
        suma += convol - events[i]*np.log(convol)
    return 2*suma

fig = plt.figure()
plt.hist(x_range, bins = 200, weights = osc_pred)
sumy = 0
for i in range(200):
    sumy += 0.05*osc_pred[i]
#print(sumy)
sigma_values = np.linspace(0.02,0.04,5)
for sig in sigma_values:
    y = convolution(x_range, osc_pred, e_range, sig)
    plt.plot(x_range, y, label = f"sigma: {sig}")
    #print(np.trapz(y, x_range))

plt.rcParams.update({'font.size': 12})
plt.xlabel(r"$Energy [GeV]$")
plt.ylabel("Reconstructed oscillated event prediction")
plt.legend()

fig = plt.figure()

def nll_alpha(bin_number, events, unosc_pred, alpha):
    suma = 0
    for i in range(bin_number):
        osc_pred = osc_prob(e_range[i], 0.777, 2.338)*unosc_pred[i]
        new_pred = osc_pred * alpha * e_range[i]
        suma += new_pred - events[i]*np.log(new_pred)
    return 2*suma


alpha_range = np.linspace(0.01, 4, 200)
plt.plot(alpha_range, nll_alpha(200, data, pred, alpha_range))
plt.xlabel("Cross-section interaction parameter")
plt.ylabel("Negative Log Likelihood")


plt.figure()
sigma_range = np.linspace(0.005, 0.05, 100)
plt.plot(sigma_range, nll_sigma(200, data, pred, sigma_range))
plt.xlabel("Gaussian width - detector resolution")
plt.ylabel("Negative Log Likelihood")




theta_values = np.linspace((np.pi/4 - 0.7), (np.pi/4 + 0.7), 300)
mdiff_values = np.linspace(1, 4, 300)

X, Y = np.meshgrid(theta_values, mdiff_values)

def nll_2d(theta, mdiff, bin_number, events, unosc_pred):
    suma = 0
    for i in range(bin_number):
        osc_pred = osc_prob(e_range[i], theta, mdiff)*unosc_pred[i]
        suma += osc_pred - events[i]*np.log(osc_pred)
    return 2*suma


Z = nll_2d(X, Y, 200, data, pred)


plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, Z, levels=200, cmap='viridis')


cbar = plt.colorbar(contour)
cbar.set_label('Function Value')
plt.xlabel(r"$Theta_{23}$")
plt.ylabel(r"$\Delta m^2_{23}$")
plt.legend()
plt.title("Negative Log Likelihood")



def nll_1d_theta(bin_number, events, unosc_pred, theta_min, theta_max):
    nll_values = []
    u_values = np.linspace(theta_min, theta_max, 200)
    for j in range(len(u_values)):
        suma = 0
        for i in range(bin_number):
            osc_pred = osc_prob(e_range[i], u_values[j])*unosc_pred[i]
            suma += osc_pred - events[i]*np.log(osc_pred)
        nll_values.append(2*suma)
    return list(u_values), nll_values

x1, y1 = nll_1d_theta(200, data, pred, 0.5, ((np.pi/4 - 0.5) + np.pi/4))
plt.rcParams.update({'font.size': 12})
plt.figure()

plt.plot(x1, y1, "green")
plt.ylabel("Negative Log Likelihood")
plt.xlabel(r"$Theta_{23}$")

plt.show()