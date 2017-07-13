__author__ = 'Li Hao'
__version__ = '3.0'
__date__ = '05/09/2016'
__copyright__ = "Copyright 2016, PI"


from setuptools import setup

import sys
import os
__CurrentPath = os.path.dirname(__file__) or '.'
sys.path.insert(0, os.path.abspath(os.path.join(__CurrentPath, os.path.pardir)))


if __name__ == '__main__':
    RootPath = os.path.abspath(os.path.join(__CurrentPath, os.path.pardir))
    os.chdir(RootPath)
    cwd = os.getcwd()
    print 'cwd: ', cwd
    setup(name='VGL',
          version='3.0.0',
          description='Vision Geometry Library',
          url='',
          author='Li Hao',
          packages=['VisionGeometryLib'],
          zip_safe=False)
