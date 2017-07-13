"""
VGL (Vision Geometry Library) is
"""
__author__ = 'Li Hao'
__version__ = '3.0'
__date__ = '02/09/2016'
__copyright__ = "Copyright 2016, PI"


from .Core import *

from numpy.testing.nosetester import _numpy_tester


test = _numpy_tester().test
bench = _numpy_tester().bench
