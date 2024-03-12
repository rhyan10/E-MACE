import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from numpy.polynomial import chebyshev
from numpy.polynomial import legendre
from scipy.linalg import companion
from scipy.linalg import fiedler_companion
from scipy.linalg import eigh
from numpy.polynomial.chebyshev import chebroots # this uses chebcompanion matrix
import time
from copy import deepcopy
from scipy.linalg import toeplitz
from numpy.fft import fft, ifft, ifftshift
import random
from scipy.signal import savgol_filter

# Function to construct intersecting cosine curves
def intersecting_cosine_curves(x):
    y1 = np.sin(x)
    y2 = np.cos(2*x)
    y3 = np.sin(2*x)
    return y1, y2, y3

# Generate x values
a = 0
b = 3
L = 1000
# x_values = np.linspace(a,b, L)  # uniform
np.random.seed(42) # make things reproducible
x_values = np.sort(np.random.uniform(a,b,L))  # random uniform dist

# Construct cosine curves
y1_values, y2_values, y3_values = intersecting_cosine_curves(x_values)

# for i in range(len(x_values)):
#     print(y3_values[i])
#     print(y2_values[i])
#     print(i)

# Separate into two curves: smaller and larger
sorted_curves = np.partition(np.vstack([y1_values, y2_values, y3_values]), [0, 1, 2], axis=0)

y1_values_o = deepcopy(sorted_curves[0, :]) 
y2_values_o = deepcopy(sorted_curves[1, :])
y3_values_o = deepcopy(sorted_curves[2, :])

y1_values = sorted_curves[0, :]
y2_values = sorted_curves[1, :] 
y3_values = sorted_curves[2, :] 

# Compute sum and multiplication of the two curves from values
sum_curve1 = y1_values + y2_values + y3_values
product_curve1 = y1_values * y2_values + y1_values * y3_values + y2_values * y3_values
product_curve2 = y1_values * y2_values * y3_values

# Interpolate with Chebyshev polynomials
degree = 20 # Adjust the degree of the polynomial

rcondf = 1e-16
# Interpolate the original curves
# from numpy.polynomial.chebyshev import chebfit as polyfit
# from numpy.polynomial.chebyshev import chebval as polyval
from numpy.polynomial.chebyshev import chebfit as polyfit
from numpy.polynomial.chebyshev import chebval as polyval
coefficients_y1 = polyfit(x_values, y1_values_o, degree, rcond = rcondf)
interpolated_y_y1 = polyval(x_values, coefficients_y1)

coefficients_y2 = polyfit(x_values, y2_values_o, degree, rcond = rcondf)
interpolated_y_y2 = polyval(x_values, coefficients_y2)

coefficients_y3 = polyfit(x_values, y3_values_o, degree, rcond = rcondf)
interpolated_y_y3 = polyval(x_values, coefficients_y3)

# Interpolate the sum and product curves
coefficients_sum = polyfit(x_values, sum_curve1, degree, rcond = rcondf)
interpolated_sum_curve1 = polyval(x_values, coefficients_sum)

coefficients_product = polyfit(x_values, product_curve1, degree, rcond = rcondf)
interpolated_product_curve1 = polyval(x_values, coefficients_product)

coefficients_product2 = polyfit(x_values, product_curve2, degree, rcond = rcondf)
interpolated_product_curve2 = polyval(x_values, coefficients_product2)

# Interpolate the cheb-ESP curves
coefficients_cheba = polyfit(x_values, -2*sum_curve1, degree, rcond = rcondf)
cheb_coeffa = polyval(x_values, coefficients_cheba)

coefficients_chebb = polyfit(x_values, 4*product_curve1+3 , degree, rcond = rcondf)
cheb_coeffb = polyval(x_values, coefficients_chebb)

coefficients_chebc = polyfit(x_values, -4*product_curve2-2*sum_curve1, degree, rcond = rcondf)
cheb_coeffc = polyval(x_values, coefficients_chebc)

# v = []
# for m in range(10000): 
#     xval = np.sort(np.random.uniform(a,b,50))  # random uniform dist
#     a = a+1e-5
#     b = b+1e-5
#     coff2 = polyfit(xval, np.sin(xval)*np.sin(2*xval)*np.cos(2*xval), degree, rcond = rcondf)
#     v.append(coff2)

def one_if_zero(vector):
    return np.where(vector == 0, 1, vector)
def crossing_tolerance(vector, tol):
    return np.where(vector < tol, 0, vector)
def fliptopos(vector):
    return np.where(vector < 0, -vector, vector)

def symmetriccompanion(p):
    (d, e) = symmetriccompanionbands(p,index)
    companion_matrix = np.diag(d) + np.diag(e, 1) + np.diag(e, -1)
    return companion_matrix

def deconvloop(p,p1,tikhonov,index):
    q, r = deconvolve_with_tikhonov(p,p1,tikhonov,index)
    if np.size(r) == 0:
        chk = -1
    else:
        chk = r[0]
    #if chk>0:
    #    print(index)
    #    print(r)
    if chk>0 and index<500:
        p1[-1] = p1[-1]+1/1000
        return deconvloop(p,p1,tikhonov,index+1)
    else:
        return (q,r)

def symmetriccompanionbands(p,index):
    op = p
    p = np.array(p)
    n = np.size(p) - 1
    p1 = np.polyder(p)/n
    d = []
    e = []
    for k in range(n):
        # q, r = np.polydiv(p,p1)
        tikhonov = 1e-16
        q, r = deconvolve_with_tikhonov(p,p1,tikhonov,index)
        # q, r = deconvloop(p,p1,tikhonov,1)
        if k < n-1:
            p = p1
            if r.any():
                p1 = r / r[0]
            e.append(-r[0])
        d.append(-q[-1])
    # note: The following are dangerous post-processing options!
    e = np.sqrt(crossing_tolerance(np.real(e[0:n-1]),1e-18)) # 0 tolerance
    d = np.array(d)
    return (d, e)


points1 = []
upper_points = []
lower_points = []
noiselevel = 0.01
peturb1 = noiselevel#*0.8891237918#*(2*random.random()-1)
peturb2 = noiselevel#*0.712938173#*(2*random.random()-1)
peturb3 = noiselevel#*0.38981492814#*(2*random.random()-1)

import torch
import torch
from torch.linalg import eigvals

def chebcompanion(c):
    one = torch.tensor([1], device=c.device)
    c = torch.cat((c, one))
    n = len(c) - 1
    mat = torch.zeros((n, n), device=c.device, dtype=c.dtype)
    scl = torch.tensor([1.] + [torch.sqrt(torch.tensor(0.5))]*(n-1), device=c.device)
    top = mat.view(-1)[1::n+1]
    bot = mat.view(-1)[n::n+1]
    top[0] = torch.sqrt(torch.tensor(0.5, device=c.device))
    top[1:] = 1/2
    bot[...] = top
    mat[:, -1] -= (c[:-1]/c[-1])*(scl/scl[-1])*0.5
    return mat

def calc_roots(c):
    a = chebcompanion(c).flip(0).flip(1)
    L = eigvals(a).real 
    sorted_L, _ = torch.sort(L)
    return sorted_L

def perturbchebroots(v,index):
    roots = calc_roots(torch.tensor(v))
    roots = roots.numpy()
    print(roots)
    return roots 

    
for i, (s,p1,p2) in enumerate(zip(cheb_coeffa, cheb_coeffb, cheb_coeffc)):
    # matrix = np.array([[0, 0, p2], [1, 0, -p1], [0, 1, s]])
    # matrix = companion([1,-s,p1,-p2]) # scipy companion matrix
    # matrix = fiedler_companion([1,-s,p1,-p2]) # fiedler banded companion matrix   
    # matrix = symmetriccompanion([1,-s,p1,-p2]) # Symmetric companion matrix by Schmeisser 1993
    # roots = np.sort(np.linalg.eigvals(matrix)) # usual eigvals
    # roots = np.linalg.eigvalsh(matrix) # symmetric eigvals from numpy
    roots = perturbchebroots([p2,p1,s],i)
    upper = perturbchebroots([p2+peturb1,p1+peturb2,s+peturb3],i)
    lower = perturbchebroots([p2-peturb1,p1-peturb2,s-peturb3],i)
    if i == 190:
        ss = s
        pp1 = p1
        pp2 = p2
    points1.append(roots)
    upper_points.append(upper)
    lower_points.append(lower)
    
# for i, (s,p1,p2) in enumerate(zip(interpolated_sum_curve1, interpolated_product_curve1, interpolated_product_curve2)):
#     # matrix = np.array([[0, 0, p2], [1, 0, -p1], [0, 1, s]])
#     # matrix = companion([1,-s,p1,-p2]) # scipy companion matrix
#     # matrix = fiedler_companion([1,-s,p1,-p2]) # fiedler banded companion matrix   
#     # matrix = symmetriccompanion([1,-s,p1,-p2]) # Symmetric companion matrix by Schmeisser 1993
#     # roots = np.sort(np.linalg.eigvals(matrix)) # usual eigvals
#     # roots = np.linalg.eigvalsh(matrix) # symmetric eigvals from numpy
#     (d1, e1) =  symmetriccompanionbands([1,-(s+peturb1),(p1+peturb2),-(p2+peturb3)],i)
#     (d2, e2) =  symmetriccompanionbands([1,-(s-peturb1),(p1-peturb2),-(p2-peturb3)],i)
#     (d, e) =  symmetriccompanionbands([1,-s,p1,-p2],i)
#     if i == 85:
#         ss = s
#         pp1 = p1
#         pp2 = p2
#         M1 = [d,e]
#         M2 = [d1,e1]
#         M3 = [d2,e2]
#     peturb = 0
#     #print(peturb)
#     # (d1,e1) = (d+peturb, e)
#     # (d2,e2) = (d-peturb, e)
#     # (d1,e1) = (d, e+peturb)
#     # (d2,e2) = (d, e-peturb)
#     roots = sp.linalg.eigvalsh_tridiagonal(d, e, lapack_driver='auto', check_finite=False)
#     upper = sp.linalg.eigvalsh_tridiagonal(d1, e1, lapack_driver='auto', check_finite=False)
#     lower = sp.linalg.eigvalsh_tridiagonal(d2, e2, lapack_driver='auto', check_finite=False)
#     points1.append(roots)
#     upper_points.append(upper)
#     lower_points.append(lower)

points1 = np.array(points1).T
upper_points = np.array(upper_points).T
lower_points = np.array(lower_points).T
diff1 = np.abs(points1[0, :] - y1_values_o)
diff2 = np.abs(points1[1, :] - y2_values_o)
diff3 = np.abs(points1[2, :] - y3_values_o)
diff = diff1 + diff2 + diff3
print(np.sum(diff))
diff1 = np.abs(upper_points[0, :] - y1_values_o)
diff2 = np.abs(upper_points[1, :] - y2_values_o)
diff3 = np.abs(upper_points[2, :] - y3_values_o)
diff = diff1 + diff2 + diff3
print(np.sum(diff))
diff1 = np.abs(lower_points[0, :] - y1_values_o)
diff2 = np.abs(lower_points[1, :] - y2_values_o)
diff3 = np.abs(lower_points[2, :] - y3_values_o)
diff = diff1 + diff2 + diff3
print(np.sum(diff))

# Plot the original curves, interpolated curves, sum, and product
plt.figure(figsize=(30, 30))

# # Original curves
# plt.subplot(5, 2, 1)
# plt.plot(x_values, y1_values_o, color='blue')
# plt.plot(x_values, y2_values_o, color='green')
# plt.plot(x_values, y3_values_o, color='red')
# plt.title('Curves')
# plt.xlabel('x')
# plt.ylabel('y')

# # Interpolated curves
# plt.subplot(5, 2, 2)
# plt.plot(x_values, interpolated_y_y1, color='blue')
# plt.plot(x_values, interpolated_y_y2, color='green')
# plt.plot(x_values, interpolated_y_y3, color='red')
# plt.title('Interpolated Curves from 1000 points')
# plt.xlabel('x')
# plt.ylabel('y')

# # Sum of curves
# plt.subplot(5, 2, 3)
# plt.plot(x_values, product_curve1, color='purple')
# plt.title('Product Curve 1')
# plt.xlabel('x')
# plt.ylabel('y')

# # Sum of curves
# plt.subplot(5, 2, 4)
# plt.plot(x_values, product_curve2, color='purple')
# plt.title('Product Curve 2')
# plt.xlabel('x')
# plt.ylabel('y')

# # Product of curves
# plt.subplot(5, 2, 5)
# plt.plot(x_values, np.absolute(interpolated_product_curve1-product_curve1), color='orange')
# plt.title('Product Curve 1 interpolation error')
# plt.xlabel('x')
# plt.ylabel('y')

# plt.subplot(5, 2, 6)
# plt.plot(x_values, np.absolute(interpolated_product_curve2-product_curve2), color='orange')
# plt.title('Product Curve 2 interpolation error')
# plt.xlabel('x')
# plt.ylabel('y')

# Reconstructed curves
plt.subplot(1, 1, 1)
plt.plot(x_values, y1_values_o, color='blue', linewidth=4)
plt.plot(x_values, y2_values_o, color='green', linewidth=4)
plt.plot(x_values, y3_values_o, color='red', linewidth=4)
plt.plot(x_values, points1[0], '--', color='brown', linewidth=4)
plt.plot(x_values, points1[1], '--' , color='brown', linewidth=4)
plt.plot(x_values, points1[2], '--', color='brown', linewidth=4)
plt.plot(x_values, upper_points[0], '--', color='purple', linewidth=4)
plt.plot(x_values, upper_points[1], '--' , color='purple', linewidth=4)
plt.plot(x_values, upper_points[2], '--', color='purple', linewidth=4)
plt.plot(x_values, lower_points[0], '--', color='black', linewidth=4)
plt.plot(x_values, lower_points[1], '--' , color='black', linewidth=4)
plt.plot(x_values, lower_points[2], '--', color='black', linewidth=4)
plt.title('Reconstructed curves from from 1000 points', fontsize=40)
plt.yticks(fontsize=40)
plt.xticks(fontsize=40)
plt.xlabel('x', fontsize=40)
plt.ylabel('y', fontsize=40)

# # Sum curve errors
# plt.subplot(5, 2, 8)
# plt.plot(x_values, np.absolute(interpolated_sum_curve1-sum_curve1), color='purple')
# plt.title('Sum Curve interpolation error')
# plt.xlabel('x')
# plt.ylabel('y')


plt.tight_layout()
plt.savefig('test.pdf') 
plt.show() 
