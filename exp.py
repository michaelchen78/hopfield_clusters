import matplotlib.pyplot as plt
from lmfit.models import SineModel, DampedHarmonicOscillatorModel, PolynomialModel, ExponentialModel
from matplotlib.pyplot import show

import numpy as np
from numpy import exp, loadtxt, pi, sqrt
from numpy import exp, sin, linspace, random
import scipy
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import leastsq, curve_fit
import lmfit
from lmfit import minimize, Parameters, Parameter
from lmfit import Model
from lmfit import models

''' DATA '''
n = np.asarray([25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37])
# eta = n/20
y1 = np.asarray([2.666666667, 7.166666667, 7.666666667, 31.33333333, 28.66666667, 56.33333333, 56.66666667,
                 133.6666667, 228.6666667, 476.5, 501.3333333, 998.1666667, 1509.5])
# eta = 20
y2 = np.asarray([2.2, 3.4, 11.2, 13.4, 34.6, 43.6, 66.4, 194.2, 355.2, 394, 445, 929.6, 1343.25])

# exponential fit
eModel = ExponentialModel()
eParams1 = eModel.guess(y1, x=n)
eResult1 = eModel.fit(y1, eParams1, x=n)
print("1", eResult1.fit_report())
eParams2 = eModel.guess(y2, x=n)
eResult2 = eModel.fit(y2, eParams1, x=n)
print("2", eResult2.fit_report())

plt.plot(n, y1, 'o', label='1 eta=n/20', color='red')
plt.plot(n, y2, 'o', label='2 eta=20', color='blue')
plt.plot(n, eResult1.init_fit, '--', label='initial fit 1', color='red')
plt.plot(n, eResult1.best_fit, '-', label='best fit 1', color='red')
plt.plot(n, eResult2.init_fit, '--', label='initial fit 2', color='blue')
plt.plot(n, eResult2.best_fit, '-', label='best fit 2', color='blue')
plt.legend()
plt.show()
plt.close()
