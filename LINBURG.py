'''
Created on 21 mar 2024
@author: Bruno Pavani Bertolino, Sofia Gemmi dos Santos
'''

from L1BURG import L1BURG
import numpy as np # Mathematical library

def LINBURG(x, log, alphamax, arquivo, topo, base, ordem, tipo_log, top, step):

    # Create a matrix of zeros.
    matrix_coef = []

    # Create the matrix which multiplies the coefficients (c.f. l1 Trend Filtering by Kim et al.)
    for i in range(len(log)):
        row = np.arange(i+1, 0, -1)
        row.resize(len(log))
        row[0] = 1
        matrix_coef.append(row)

    matrix_coef = np.asarray(matrix_coef)

    # Create the regularized (L1) input for the Burg algorithm
    z = {}
    z['0'] = np.asarray(log)

    # Call L1BURG four times: each one linearizes the last output again
    for i in range(1,5):
        z['{0}'.format(i)] = L1BURG(arquivo = arquivo, topo = topo, base = base, ordem = ordem, \
                                    tipo_log = tipo_log, log = z['{0}'.format(i-1)], \
                                    alpha = 10**(-4+i)*alphamax, matrix = matrix_coef)

    # At the end, compute an average of the linearized signals and deduct from the original z['0']
    reg_data_comp = z['0'] - (z['1'] + z['2'] + z['3'] + z['4'])/4.0

    # Burg algorithm for i = 0 and i = 1
    den = np.zeros(ordem+1)
    k = np.zeros(ordem+1)

    f = np.zeros((ordem+1,len(x)+1))
    b = np.zeros((ordem+1,len(x)+1))
    fr = np.zeros((ordem+1,len(x)))
    br = np.zeros((ordem+1,len(x)))

    # Initialize the forward and backward prediction vectors with the data from before
    for p in range(len(x)):
        f[0, p] = reg_data_comp[p]
        b[0, p] = reg_data_comp[p]

    for p in range(len(x)-1):
        fr[0, p] = f[0, p+1]
        br[0, p] = b[0, p]

    k[0] = -2 * np.inner(br[0],fr[0]) / (np.linalg.norm(fr[0])**2 + np.linalg.norm(br[0])**2)
    den[0] = np.linalg.norm(fr[0])**2 + np.linalg.norm(br[0])**2

    for p in range(len(x)):
        f[1, p] = fr[0, p] + k[0]*br[0, p]
        b[1, p] = br[0, p] + k[0]*fr[0, p]

    for p in range(len(x)-1):
        fr[1, p] = f[1, p+1]
        br[1, p] = b[1, p]

    k[1] = -2 * np.inner(br[1],fr[1]) / (np.linalg.norm(fr[1])**2 + np.linalg.norm(br[1])**2)
    den[1] = np.linalg.norm(fr[1])**2 + np.linalg.norm(br[1])**2


    # Burg algorithm for i > 1
    for i in range(1, ordem):

        for p in range(len(x)-i-1):
            fr[i, p] = f[i, p+1]
            br[i, p] = b[i, p]

        k[i] = -2 * np.inner(br[i],fr[i]) / den[i]

        if i >= ordem:
            break

        for p in range(len(x)-i-1):
            f[i+1, p] = fr[i, p] + k[i]*br[i, p]
            b[i+1, p] = br[i, p] + k[i]*fr[i, p]

        den[i+1] = (1-k[i]**2)*den[i] - (f[i+1, 0])**2 - (b[i+1, len(x)-i-1])**2

    # Creates an "erro" (meaning prediction error) variable equaling minus the forward prediction vector.
    erro = np.zeros(len(x))

    for p in range(0, len(x)-ordem):
        if p == len(x)-ordem-1:
            break
        erro[p] = -f[ordem, p]

    for p in range(len(x)-ordem, len(x)):
        erro[p] = -b[ordem, p]

    # Cumulative integration within a window of size "ordem"
    integral_burg = np.zeros(len(x))
    for i in range(int((topo-top)/step)-ordem, len(x)-ordem):
        if i == len(x)-ordem-1:
            break
        integral_burg[i] = -(np.sum(erro[i:i+ordem]))
    for i in range(len(x)-ordem, len(x)):
        integral_burg[i] = -(np.sum(erro[i-ordem:i]))

    integral_burg_norm = integral_burg / max(abs(integral_burg))

    return integral_burg_norm

