'''
Created on 21 mar 2024
@author: Bruno Pavani Bertolino, Sofia Gemmi dos Santos
'''

from sklearn import linear_model # Library that contains a Lasso regularization algorithm

def L1BURG(arquivo, topo, base, ordem, tipo_log, log, alpha, matrix):

    # Regularizes according to an L1 (Lasso) method.
    fitted = linear_model.Lasso(alpha, precompute = True, max_iter=50)
    fitted.fit(matrix, log)
    coefs = fitted.coef_
    reg_data = matrix.dot(coefs)

    return reg_data
