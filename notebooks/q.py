import numpy as np

E = 588420
D = 49330
net_D = 28224
rfr = 0.04091
beta = 0.57
erp = 0.06
r_e = rfr + erp * beta
interest_exp = 1 / 42.30
WACC = (E * r_e / (E + D)) + (D * interest_exp * (1 - 0.177) / (E + D))
print(WACC)
FCF = [22277.7, 24179.2, 26493.5, 28890.1, 31577.6]

g = 0.026
TV = (FCF[4] * (1 + g) / (WACC - g)) / (1 + WACC) ** 4.83
print(TV)
PV = []

for i in range(len(FCF) - 1):
    PV_n = FCF[i] / (1 + WACC) ** (i + 0.83)
    PV.append(PV_n)

PV = np.array(PV)
EV = TV + np.sum(PV)
equity = EV - net_D
intrinsic = equity / 2410
print('The intrinsic value per share is $', intrinsic)