# Kalman-filter-shear-wave-tracking

**General description**
  - Given: A space time map of a shear wave propagation (harmonic vibrations) measured with noticeable additive white Gaussian noise (here: simulated signal)
  - Aim: (1) Estimate the true amplitude & phase of the harmonic vibration from the measured signals with the least mean square error via Kalman filter (2) track & calculate the shear wave propagation speed to infer the stiffness (Young's modulus) of the sample 


**Mathmetical description (Kalman filter)**

In general case:
 - Observed data: y<sub>k</sub> = H<sub>k</sub> * x<sub>k</sub> + n<sub>k</sub> (where n<sub>k</sub> = noise)
 - Aim: Estimate state variables at the k+1 th step  x<sub>k+1</sub> =  phi * x<sub>k</sub> + w<sub>k</sub> (where w<sub>k</sub>: noise)
 - Method: Minimize the estimation mean square error: P<sub>k</sub> = E[(x<sub>k</sub>-x<sub>k</sub>_hat)(x<sub>k</sub>-x<sub>k</sub>_hat)']

In our particular example (harmonic vibration): 
- Measured signal is modeled as: y<sub>k</sub> = beta<sub>k</sub> * cos(2*pi*f<sub>m</sub>*k*T + ph<sub>k</sub>) + n<sub>k</sub>
- Aim: Estimate (1) vibration amplitude (beta<sub>k</sub>) & (2) vibration phase (ph<sub>k</sub>)
- Method: Setting our state variable:
x<sub>k</sub> = [x<sub>k</sub>(1), x<sub>k</sub>(2)]' = [beta<sub>k</sub>*cos(ph<sub>s,k</sub>), beta<sub>k</sub>*sin(ph<sub>s,k</sub>)]',  H = [cos(2*pi*k*T), -sin(2*pi*k*T)]

Kalman filter steps:
- Initialization
- Get Kalman Gain at the k-th step G<sub>k</sub> = P<sub>k</sub>_prior * H<sub>k</sub>' * (H<sub>k</sub> * P<sub>k</sub>_prior * H<sub>k</sub>' + R)<sup>-1</sup>
- State estimate update: x<sub>k</sub>_hat = x<sub>k</sub>_hat_prior + G<sub>k</sub> * (y<sub>k</sub> - H<sub>k</sub>*x<sub>k</sub>_hat_prior)
- Update the estimation error variance matrix: P<sub>k</sub> = (I-G<sub>k</sub>*H<sub>k</sub>)*P<sub>k</sub>_prior
- Project to k+1-th step: x<sub>k+1</sub>_hat_prior = phi<sub>k</sub> * x<sub>k</sub>_hat, P<sub>k+1</sub>_prior = phi<sub>k</sub> * P<sub>k</sub> * phi<sub>k</sub>' + Q

**Note**
The Kalman-filter-based shear wave amplitude & phase estimation method was inspired by [1], where [2] provides a concise and theoretical foundation of of Kalman filter. The parameters of the harmonic motion excitation, imaging system detection, and sample properties used in this code followed those provided in [3].  
Shear wave velocity calculation and its relationship to Young's modulus (stiffness measure) can be found in both [1] & [3].



**References**
- [1] Greenleaf, J. F., et al. "Detection of tissue harmonic motion induced by ultrasonic radiation force using pulse-echo ultrasound and Kalman filter." *IEEE transactions on ultrasonics, ferroelectrics, and frequency control* 54.2 (2007): 290-300
- [2] Lacey, T., "Tutorial: The Kalman Filter", http://web.mit.edu/kirtley/kirtley/binlustuff/literature/control/Kalman%20filter.pdf
- [3] Huang, P.-C., et al. "Interstitial magnetic thermotherapy dosimetry based on shear wave magnetomotive optical coherence elastography." *Biomedical optics express* 10.2 (2019): 539-551.
