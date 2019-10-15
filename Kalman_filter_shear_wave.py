import numpy as np
import matplotlib.pyplot as plt
import math

############## Generate data for simulation ##############
# sample parameters (ground truth)
E = 5e3  # Young's modulus (Pa) of the sample
mu = 0.47  # Poisson's ratio of the sample
rho = 1040  # density (kg*m^-3) of the sample
Cs = np.sqrt(E / (2 * (1 + mu) * rho))  # shear wave propagation velocity (m/s) propagating on the sample

# shear wave (excitation) & imaging (detection) parameters used
# to simulate the propagation of a shear wave oscillating at 460 Hz
# where the shear wave vibration was captured by an imaging system
fm = 460  # modulation frequency (Hz)
fs = 91912 * 0.995  # Sampling rate (Hz)
nt = 500  # number of time points
t = np.linspace(0, nt / fs, nt)  # time

sw_lambda = Cs / fm  # shear wave wavelength (m)
r = 4e-3  # propagation radius (m)
phs = 2 * np.pi * r / sw_lambda  # phase change over r (rad)
lateral_scan_range_pix = 400  # lateral scan range (pix)
dphs = phs / lateral_scan_range_pix  # phase change/lateral pixel

# create the space time map (with & without noise)
# the harmonic vibration source was generated at x=0
spacetimemap = []
spacetimemap_w_noise = []
for ix in range(lateral_scan_range_pix):
    dphase = dphs * ix  # phase delay
    harmonic_motion = np.cos(2 * np.pi * fm * t - dphase)  # harmonic motion at one x position
    spacetimemap.append(harmonic_motion)
    wgn = np.random.normal(0, 0.2, np.shape(harmonic_motion))  # additive Gaussian noise
    spacetimemap_w_noise.append(harmonic_motion + wgn)

# display
plt.figure(figsize=(12,8))
plt.subplot(121)
plt.imshow(np.transpose(spacetimemap))
plt.title("space time map (ground truth)")
plt.colorbar()
plt.xlabel("radial position (pix)"), plt.ylabel("time points")
plt.subplot(122)
plt.imshow(np.transpose(spacetimemap_w_noise))
plt.title("space time map (w. noise)")
plt.colorbar()
plt.xlabel("radial position (pix)"), plt.ylabel("time points")
plt.show()
plt.tight_layout()

ixoi = 199
plt.figure()
plt.plot(spacetimemap_w_noise[ixoi], '*-')
plt.plot(spacetimemap[ixoi])
plt.title('harmonic motion at radial position r =' + str(ixoi))
plt.legend(["w. noise", "w.o. noise (ground truth)"])
plt.xlabel("time points"), plt.ylabel("vibration amplitude (a.u.)")
plt.show()


############## Perform Kalman filter for denoising ##############
# Aim: find state variable x_(k+1) = phi * x_k + w_k, where w_k = noise
# Observed: y_k = H_k * x_k + n_k, where n_k = additive white noise
# Estimation error variance: P_k = E[(x_k-x_k_hat)(x_k-x_k_hat)']
# State estimate update: x_k_hat = x_k_hat_prior + G_k*(y_k - H_k*x_k_hat_prior)

# Here, the measured harmonic vibration signal can be modeled as:
# y_k = beta_k * cos(2*pi*fm*k*T + ph_k) + n_k
# where we want to estimate (1) vibration amplitude (beta_k) & (2) vibration phase (ph_k)
# This can be achived by setting our state variable:
# x_k = [x_k1, x_k2]' = [beta_k*cos(ph_s), beta_k*sin(ph_s)]'
# and H = [cos(2*pi*k*T), -sin(2*pi*k*T)]


# set parameters
T = 1 / fs  # temporal interval
phi = np.eye(2)  # state transition matrix
Q = np.zeros((2, 2))  # uncerntainty in the model = E[w_k * w_k']
n = np.random.normal(0, 0.01, np.shape(t))  # additive white noise
R = np.var(n)  # variance of n (as we only measure single y_k, R=E(n_k * n_k'), dim(R) = 1x1)

# perform Kalman filter at each lateral position
y_est2D = []
Amplitude_median = []
Phase_median = []
for ix in range(lateral_scan_range_pix):
    y = spacetimemap_w_noise[ix]

    # initialization
    P_prior = 1e-1 * np.eye(2)
    x_prior = np.array([[y[0]], [y[1]]])
    xk1 = []  # the cosine component to be estimated
    xk2 = []  # the sine component to be estimated
    y_est = []
    for k in range(len(y)):
        yk = y[k]

        # noise connection between state variables (x_k) & measured signal (y_k)
        H = np.array([[np.cos(2 * np.pi * fm * k * T), -np.sin(2 * np.pi * fm * k * T)]])

        # get Kalman Gain at the k-th step
        S = np.dot(H, np.dot(P_prior, np.transpose(H))) + R
        S_inv = np.linalg.inv(S)
        G = np.dot(np.transpose(P_prior), np.dot(np.transpose(H), S_inv))

        # update the state estimation
        x = x_prior + np.dot(G, (yk - np.dot(H, x_prior)))

        # update the estimation error variance of x_k
        P = np.dot((np.eye(2) - np.dot(G, H)), P_prior)

        # project to k+1
        x = np.dot(phi, x)
        P = np.dot(phi, np.dot(P, np.transpose(phi))) + Q

        # store the state variables values
        x1, x2 = x[0], x[1]
        xk1.append(x1[0])  # x1 = beta * cos(ph)
        xk2.append(x2[0])  # x2 = beta * sin(ph)

        # store the estimated y_k values (obtained from the estimated x_k)
        y_est_temp = np.dot(H, x) + n[k]
        y_est_temp = y_est_temp[0]
        y_est.append(y_est_temp[0])

        P_prior = P
        x_prior = x

    y_est2D.append(y_est)

    # get vibration amplitude from state variables
    beta = np.sqrt(np.power(xk1, 2) + np.power(xk2, 2))
    Amplitude_median.append(np.median(beta))

    # get vibration phase from state variables
    ph = np.arctan(np.divide(xk2, xk1))
    Phase_median.append(np.median(ph))



# display: spacetime map & harmonic motion
plt.figure(figsize=(12,8))
plt.subplot(121)
plt.imshow(np.transpose(spacetimemap_w_noise))
plt.title("space time map (w. noise)")
plt.colorbar()
plt.xlabel("radial position (pix)"), plt.ylabel("time points")
plt.subplot(122)
plt.imshow(np.transpose(y_est2D))
plt.title("space time map (after Kalman filter)")
plt.colorbar()
plt.xlabel("radial position (pix)"), plt.ylabel("time points")
plt.show()
plt.tight_layout()

ixoi = 199
plt.figure()
plt.plot(spacetimemap_w_noise[ixoi], '*-')
plt.plot(y_est2D[ixoi])
plt.title('harmonic motion at radial position r =' + str(ixoi))
plt.legend(["w. noise", "after Kalman filter"])
plt.xlabel("time points"), plt.ylabel("vibration amplitude (a.u.)")
plt.show()


# display: vibration amplitude & phase

# Phase unwrap by ph_median = unwrap(ph_median*2)/2
Phase_median = [i * 2 for i in Phase_median]
Phase_median = np.unwrap(Phase_median)
Phase_median = [i/2 for i in Phase_median]

plt.figure()
plt.subplot(211)
plt.plot(Amplitude_median)
plt.ylim([0 , 1.5])
plt.title("after Kalman filter: median vibration amplitude at each r position")
plt.xlabel("radial position (pix)")
plt.subplot(212)
plt.plot(Phase_median)
plt.title("after Kalman filter: median vibration phase at each r position")
plt.xlabel("radial position (pix)")
plt.tight_layout()
plt.show()


########### Calculate shear wave velocity & Young's modulus ##############
dr = r/lateral_scan_range_pix                      # radial distance per pixel (m/pix)
dr_val = np.linspace(0, r, lateral_scan_range_pix)

# linear fit the phase signal along propagation distance
sig = Phase_median
p = np.polyfit(dr_val, sig, 1)
sig_fit = dr_val*p[0] + p[1]

# display
dr_val = [i*1000 for i in dr_val]

plt.figure()
plt.plot(dr_val, sig)
plt.plot(dr_val, sig_fit,'r--')
plt.xlabel("radial distance (mm)")
plt.ylabel("phase (rad)")
plt.legend(["phase (original)", "phase (linear fit)"])
plt.show()

slope = p[0]                         # = dphase/dr = wave number
Cs_estimated = 2*np.pi*fm/slope      # estimated shear wave propagation velocity

G_estimated = rho*np.power(Cs_estimated, 2)   # estimated shear modulus
E_estimated = 2*G_estimated*(1+mu)            # estimated Young's modulus (Pa)

prc_err = 100*(E_estimated - E)/E

print("Young's modulus of the sample is estimated to be %f %s %f%s" % (E_estimated/1000, "kPa. (Error:", prc_err, "%)"))

