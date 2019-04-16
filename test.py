import unittest, importlib
from smp_base.impl import smpi

# func = {}

imports_req = {
    'numpy': 'np',
}

imports_opt = {
    'matplotlib.pyplot': 'plt',
}
        
# from cloud.serialization.cloudpickle import dumps
# from collections import OrderedDict
# from cycler import cycler
# from emd import emd
# from functools import partial
# from functools import partial, wraps
# from functools import reduce
# from igmm_cond import IGMM_COND
# from jpype import getDefaultJVMPath, isJVMStarted, startJVM, attachThreadToJVM, isThreadAttachedToJVM
# from jpype import JPackage
# from kohonen.kohonen import argsample
# from kohonen.kohonen import Gas, GrowingGas, GrowingGasParameters, Filter
# from kohonen.kohonen import Map, Parameters, ExponentialTimeseries, ConstantTimeseries
# from matplotlib.font_manager import FontManager
# from matplotlib.font_manager import FontProperties
# from matplotlib import colorbar as mplcolorbar
# from matplotlib import gridspec
# from matplotlib import rc
# from matplotlib import rcParams
# from matplotlib import rc, rcParams, rc_params
# from matplotlib.pyplot import figure
# from matplotlib.table import Table as mplTable
# from .models_reservoirs import Reservoir, Reservoir2
# from .models_reservoirs import res_input_matrix_random_sparse
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# from numpy import pi, tanh, clip, inf, cos, sin, array, poly1d, sign
# from numpy.linalg import norm
# from otl_oesgp import OESGP
# from otl_storkgp import STORKGP
# from pandas.tools.plotting import scatter_matrix
# from pickle import Pickler
# from pickle import Unpickler
# from pyemd import emd as pyemd
# from pyemd import emd_with_flow as pyemd_with_flow
# from pyunicorn.timeseries import RecurrencePlot
# from scipy import signal
# from scipy.io import wavfile
# from scipy.io import wavfile as wavfile
# from sklearn import kernel_ridge
# from sklearn import linear_model
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.preprocessing import normalize
# from sklearn.utils import shuffle
# from smp_base.common import get_module_logger
# from smp_base.common import get_module_logger, compose
# from smp_base.common import set_attr_from_dict
# from smp_base.config import RLSPY
# from smp_base.eligibility import Eligibility
# from smp_base.measures import div_kl, meas_hist
# from smp_base.measures import meas
# from smp_base.measures import measures as measures_available
# from smp_base.measures_infth import init_jpype, dec_compute_infth_soft
# from smp_base.models import iir_fo
# from smp_base.models import make_figure, make_gridspec
# from smp_base.models import savefig, plot_nodes_over_data_1d_components_fig, plot_nodes_over_data_1d_components
# from smp_base.models import smpModelInit, smpModel
# from smp_base.models import smpModelInit, smpModelStep, smpModel
# from smp_base.models_learners import smpSHL
# from smp_base.models_reservoirs import LearningRules
# from smp_base.models_reservoirs import Reservoir, LearningRules
# from smp_base.plot_utils import put_legend_out_right
# # from smp.datasets import wavdataset
# # from smp.infth import init_jpype, ComplexityMeas
# from smp_msgs.msg import reservoir
# from smptests import TestLearner
# from std_msgs.msg import Float32, Float32MultiArray
# from sympy import symbols
# #             if import_flag:
# #     #     if kwargs.has_key('import_name'):
# import argparse
# import argparse, pickle
# import argparse, sys
# import colorcet as cc
# import configparser, ast
# import copy
# #             # import_flag = eval(self.import_name)
# #             import_flag = HAVE_PYUNICORN
# import itertools
# import logging
# # import matplotlib as mpl
# import matplotlib.colors as mplcolors
# import matplotlib.gridspec as gridspec
# import matplotlib.patches as mplpatches
# import matplotlib.pylab as pl
# import matplotlib.pylab as plt
# # import mdp
# import numpy.linalg as LA
# import Oger
# import os
# import pandas as pd
# #         # import pdb; pdb.set_trace()
# import pickle
# # import pylab as pl
# import pylab as pl
# import pypr.clustering.gmm as gmm
# import rlspy
# import rospy
# import scipy as sp
# import scipy.interpolate as si
# import scipy.linalg as sLA
# import scipy.signal as ss
# import scipy.sparse as spa
# import scipy.sparse as sparse
# import scipy.stats  as stats
# import seaborn as sns
# import sklearn
# import smp_base.config as config
# import sys
# import sys, os
# import sys, time
# import sys, time, argparse
# import threading, signal, time, sys
# import time
# import unittest

# print("Cannot initialize jpype because it couldn't be imported. Make sure jpype is installed")
# print("Couldn't import emd from emd with %s, make sure emd is installed." % (e, ))
# print("Couldn't import emd from pyemd with %s, make sure pyemd is installed." % (e, ))
# print("Couldn't import IGMM lib", e)
# print("Couldn't import init_jpype from measures_infth, make sure jpype is installed", e)
# print("Couldn't import jpype, %s" % e)
# print("Couldn't import lmjohns3's kohonon SOM lib", e)
# print("couldn't import online GP models:", e)
# print("Couldn't import pypr.clustering.gmm", e)
# print("Couldn't import RecurrencePlot from pyunicorn.timeseries, make sure pyunicorn is installed", e)
# print("Couldn't import seaborn as sns, make sure seaborn is installed", e)

# def test_imports():
#     assert True, 'Seriously broken, no idea'

class Testimpl(unittest.TestCase):

    def test_smpi(self):
        self.assertEqual(smpi('numpy'), importlib.import_module('numpy'), "Should be numpy")

    # def test_sum_tuple(self):
    #     self.assertEqual(sum((1, 2, 2)), 6, "Should be 6")

if __name__ == '__main__':
    unittest.main()
