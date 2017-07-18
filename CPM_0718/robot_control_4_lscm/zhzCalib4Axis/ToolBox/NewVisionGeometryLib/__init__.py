"""
VGL (Vision Geometry Library) is a library that includes several functions of vision project.
    It contains:
        1. Coordinate transformation.
        2. Camera geometry.
        3. Other functions...
"""
__author__ = 'Li Hao'
__version__ = '3.0'
__date__ = '01/11/2016'
__copyright__ = "Copyright 2016, PI"


from .Core import *

from numpy.testing.nosetester import NoseTester


test = NoseTester().test
bench = NoseTester().bench
