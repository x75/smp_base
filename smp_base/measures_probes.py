"""smp_base.measures_probes

.. moduleauthor:: Oswald Berthold, 2018

Probe style measures
"""

import numpy as np

import sklearn
from sklearn import linear_model
from sklearn import kernel_ridge
from sklearn.model_selection import train_test_split

def meas_linear_regression_probe(data, alpha = 0.0):
    """meas_linear_regression_probe

    Linear regression probe: evaluate random vector with respect to
    regression task
    """
    assert 'X' in data and 'Y' in data, "Data requires 'X' and 'Y' keys with input and labels resp."

    # linear model
    lm = linear_model.Ridge(alpha = alpha)

    # data
    X_train, X_test, y_train, y_test = train_test_split(data["X"], data["Y"], random_state=1)

    # debug plot
    # pl.subplot(211)
    # pl.plot(data["Y"])
    # pl.subplot(212)
    # pl.plot(range(y_train.shape[0]), y_train)
    # pl.plot(range(y_train.shape[0], y_train.shape[0]+ y_test.shape[0]), y_test)
    # pl.show()
    
    # lm.fit(data["X"], data["Y"])
    # Y_ = lm.predict(data["X"]) # training error
    # mse = np.mean(np.square(data["Y"] - Y_))

    # fit the model
    lm.fit(X_train, y_train)
    # test the model
    y_ = lm.predict(X_test)
    # compute measure
    mse = np.mean(np.square(y_test - y_))
    w_norm = np.linalg.norm(lm.coef_)
    intercept_norm = np.linalg.norm(lm.intercept_)
    n_iter = lm.n_iter_

    print "w_norm = %s, intercept_norm = %s, n_iter_ = %s" % (w_norm, intercept_norm, n_iter)
    
    # print "regression training MSE = %f" % (mse)

    # # pl.plot(data["Y"])
    # # pl.plot(Y_)

    # # get index for y_test sorted
    # idx = np.argsort(y_test, axis=0)
    # print y_test.shape, idx.shape
    # print "idx", idx, idx.flatten()

    # # get sorted data
    # y_sorted = y_[idx.flatten()]
    # print "y_sorted", y_sorted.shape

    # # new mode
    # lm2 = linear_model.Ridge(alpha=0.0)
    # y_sorted_flat = y_sorted.copy() # .reshape((-1, 1))
    # idx_flat = np.arange(y_sorted.shape[0]).reshape((-1, 1))
    # print "shapes y_sorted_flat, idx_flat", y_sorted_flat.shape, idx_flat.shape
    # # fit y to indices
    # lm2.fit(idx_flat, y_sorted_flat)
    # # print dir(lm2)
    # print lm2.coef_, lm2.intercept_

    # # another mode, testing kernel ridge
    # krr = kernel_ridge.KernelRidge(alpha = 0.0, gamma = 0.01, kernel="rbf")
    # krr.fit(X_train, y_train)
    # y_krr = krr.predict(X_test)
    # y_krr_sorted = y_krr[idx.flatten()]
    
    # # pl.plot(y_test[idx.flatten()])
    # # pl.plot(y_sorted_flat)
    # # pl.plot(y_krr_sorted)
    # # # pl.plot(idx_flat, lm2.coef_ * idx_flat.T + lm2.intercept_)
    # # pl.show()

    # shape match prediction
    y_ = lm.predict(data['X'])
    return y_, mse
        
