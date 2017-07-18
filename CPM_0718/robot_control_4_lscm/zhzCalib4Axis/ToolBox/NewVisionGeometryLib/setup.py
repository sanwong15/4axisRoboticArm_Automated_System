__author__ = 'Li Hao'
__version__ = '3.0'
__date__ = '02/09/2016'
__copyright__ = "Copyright 2016, PI"


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('VisionGeometryLib', parent_package, top_path)
    config.add_data_dir('Tests')
    config.name = 'VGL'
    config.version = '3.0.0'
    config.description='Vision Geometry Library'
    config.url=''
    config.author='Li Hao'
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(configuration=configuration)
