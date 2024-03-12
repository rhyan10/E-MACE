import torch

def polyder(coeffs):
    # Compute derivative of polynomial coefficients
    if len(coeffs) == 1:
        return torch.tensor([0.], dtype=coeffs.dtype)
    else:
        deriv_coeffs = torch.arange(len(coeffs) - 1, 0, -1, dtype=coeffs.dtype, device=coeffs.device) * coeffs[:-1]
        return deriv_coeffs

def symmetriccompanion(p):
    (d, e) = symmetriccompanionbands(p)
    companion_matrix = torch.diag(d) + torch.diag(e, 1) + torch.diag(e, -1)
    return companion_matrix

def crossing_tolerance(vector, tol):
    return torch.where(vector < tol, torch.tensor(0), vector)

def tridiagonal_matrix(n, diag_vals, offdiag_vals):
    return torch.diag(diag_vals) + \
           torch.diag(offdiag_vals, diagonal=-1) + \
           torch.diag(offdiag_vals, diagonal=1)

def polydiv(dividend, divisor):
    """
    Perform polynomial division of two tensors.

    Parameters:
        dividend (torch.Tensor): The coefficients of the dividend polynomial.
        divisor (torch.Tensor): The coefficients of the divisor polynomial.

    Returns:
        tuple(torch.Tensor, torch.Tensor): Quotient and remainder polynomials.
    """
    dividend_deg = len(dividend) - 1
    divisor_deg = len(divisor) - 1

    quotient_deg = dividend_deg - divisor_deg

    if quotient_deg < 0:
        # If dividend degree is less than divisor degree, return quotient as zero
        return torch.zeros_like(dividend), dividend

    quotient = torch.zeros(quotient_deg + 1, dtype=dividend.dtype, device=dividend.device)
    remainder = dividend

    divisor_shifted = torch.zeros_like(dividend)
    divisor_shifted[:len(divisor)] = divisor

    for i in range(quotient_deg + 1):
        d = torch.zeros(len(divisor_shifted), device=divisor_shifted.device)
        d[0] = 1
        d = torch.sum(d * divisor_shifted)
        quotient_mask = torch.zeros(len(quotient), device=remainder.device)
        quotient_mask[i] = 1
        quotient += (quotient_mask*remainder[:len(quotient)]) / d
        mask = torch.zeros(len(divisor_shifted), device=divisor_shifted.device)
        mask[:len(divisor)] = 1

        quotient_mask = quotient * quotient_mask
        sum_quotient = torch.sum(quotient_mask)

        rem = sum_quotient * (divisor_shifted*mask)
        rem = torch.roll(rem, i, 0)
        rem_mask = torch.zeros(len(remainder), device=divisor.device)
        rem_mask[i:i+len(divisor)] = 1
        rem = rem * rem_mask
        remainder -= rem
        
    remainder = remainder[remainder!=0]
    return quotient, remainder

def polyval(p, x):
    result = torch.zeros_like(x, device=p.device)
    degree = p.size(0) - 1
    for i in range(degree + 1):
        result += p[i] * (x ** (degree - i))
    return result

def symmetriccompanionbands(p):
    n = p.size()[0] - 1
    p1 = polyder(p) / n
    d = []
    e = []
    for k in range(n):
        q, r = polydiv(p, p1)
        s = -polyval(r.flip(0), torch.tensor(0.)) / polyval(p1.flip(0), torch.tensor(0.))
        if k < n:
            p = p1
            p1 = polyder(p1)
            if r.any():
                p1 = -r / s
        d.append(-q[-1])
        e.append(s)
    d = torch.stack(d)
    e = torch.stack(e[:-1])
    return d, e

def calc_roots(s, p1, p2):
    concatenated_tensor = torch.stack((torch.tensor(1, device=s.device), -s, p1, -p2))
    (d, e) = symmetriccompanionbands(concatenated_tensor)
    matrix = tridiagonal_matrix(3, d, e)
    roots, _ = torch.linalg.eig(matrix)
    return roots.real