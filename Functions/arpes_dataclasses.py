"""
Class Definitions for ARPES Data (2D and 3D)
@author: Alexandra Tully
@date: November 2020
"""


import os
import numpy as np
# from Data2DObject import AaData2D, AaData3D, UBCGold
from zwartsenberg.zwartsenberg_data import AaData2D, AaData3D
from functools import lru_cache


Path = 'C:/Users/atully/Code/ARPES Code Python/analysis_data/'


class BaseData:
    def __init__(self, fp, light_source, cryo_temp, filename):
        self.fp = fp
        self.laser = light_source
        self.cryo_temp = cryo_temp
        self.filename = filename


class Data2D(BaseData):
    def __init__(self, fp, light_source, cryo_temp, filename):  # inherits everything from BaseData, with the option to overwrite
        super().__init__(fp, light_source, cryo_temp, filename)  # super is one up (BaseData)
        data = AaData2D(self.fp, datatype='igor')
        self.berend_dataclass = data
        self.data = data.data  # the array of data
        self.xaxis = data.yaxis
        self.yaxis = data.xaxis  # Berend's xaxis is E, but he named them wrong: np naming convention is [z, y, x]
        self.energy_center = np.mean(data.xaxis)
        self.energy_low = data.xaxis[0]
        self.energy_high = data.xaxis[-1]

    @classmethod
    @lru_cache(maxsize=100)
    # def single_load(cls, month, year=2020, laser='lamp', cryo_temp='RT', scan_type='HS', k_cut=None, scan_number=1, filename=None,
    #                 filepath=Path):
    def single_load(cls, month, year=2021, light_source='Lamp', cryo_temp='LT', scan_number=1, filename=None, filepath=Path):
        ddir = os.path.normpath(filepath)
        fp = os.path.join(ddir, f'{month}_{year}', cryo_temp, light_source, '2D')
        if not filename:
            # filename = cls._get_filename(cryo_temp, scan_type, k_cut, scan_number)
            filename = cls._get_filename(light_source, scan_number)
        data_path = os.path.join(fp, filename)
        return cls(data_path, light_source, cryo_temp, filename)

    @staticmethod  # normal function but only associated with Data2D class
    # def _get_filename(cryo_temp, scan_type, k_cut, scan_number):
    #     if scan_type == 'HS':
    #         if k_cut not in ['GM', 'KG']:
    #             raise ValueError(f'{k_cut} is not an expected k_cut')
    #         if cryo_temp == 'RT':
    #             fn = f'HS_{k_cut}{scan_number:04d}_{scan_number:03d}.ibw'
    #         elif cryo_temp != 'RT':
    #             fn = f'HS_{k_cut}_{cryo_temp}{scan_number:04d}_{scan_number:03d}.ibw'
    #         else:
    #             raise NotImplementedError
    #     elif scan_type == 'UPS':
    #         if cryo_temp == 'RT':
    #             fn = f'UPS_{scan_number:04d}_{scan_number:03d}.ibw'
    #         elif cryo_temp != 'RT':
    #             fn = f'UPS_{cryo_temp}{scan_number:04d}_{scan_number:03d}.ibw'
    #         else:
    #             raise NotImplementedError
    #     elif scan_type == 'Raster':
    #         fn = f'Raster{scan_number:04d}_{scan_number:03d}.ibw'
    #     else:
    #         raise ValueError(f'get_filename with ({cryo_temp, scan_type, k_cut, scan_number}) failed')
    #     return fn
    def _get_filename(light_source, scan_number):
        fn = f'OMBE_{light_source}_2D{scan_number:04d}_.ibw'
        return fn


class Data3D(BaseData):
    def __init__(self, fp, light_source, cryo_temp, filename):
        super().__init__(fp, light_source, cryo_temp, filename)
        data = AaData3D(self.fp, datatype='APE', zaxis='theta_y')
        self.berend_dataclass = data
        self.data = data.data  # the array of data (3D cube)
        self.xaxis = data.zaxis
        self.yaxis = data.yaxis
        self.zaxis = data.xaxis
        self.energy_center = np.mean(data.xaxis)
        self.energy_low = data.xaxis[0]
        self.energy_high = data.xaxis[-1]

    @classmethod
    # def single_load(cls, month, year=2020, laser='lamp', cryo_temp='RT', scan_type='FS', scan_subtype=None,
    #                 scan_number=1, filename=None, filepath=Path):
    def single_load(cls, month, year=2021, light_source='Lamp', cryo_temp='LT', scan_type='deflector_scan', scan_number=1,
                    filename=None, filepath=Path):
        # if scan_subtype is None:
        #     raise ValueError(f'No scan subtype filename provided')
        ddir = os.path.normpath(filepath)
        fp = os.path.join(ddir, f'{month}_{year}', cryo_temp, light_source, '3D', scan_type)
        if not filename:
            filename = cls._get_filename(light_source, scan_type, scan_number)
        data_path = os.path.join(fp, filename)
        return cls(data_path, light_source, cryo_temp, filename)

    @classmethod
    # def single_load(cls, month, year=2020, laser='lamp', cryo_temp='RT', scan_type='FS', scan_subtype=None,
    #                 scan_number=1, filename=None, filepath=Path):
    def single_load_2022(cls, filepath, filename, light_source='XUV', cryo_temp='LT'):
        # if scan_subtype is None:
        #     raise ValueError(f'No scan subtype filename provided')
        fp = os.path.join(filepath, filename)
        return cls(fp, light_source, cryo_temp, filename)

    @staticmethod  # normal function but only associated with Data3D class
    # def _get_filename(cryo_temp, scan_type, scan_subtype, scan_number):
    #     if scan_type == 'FS':
    #         if cryo_temp == 'RT':
    #             fn = f'FS_{scan_subtype}{scan_number:04d}'
    #         elif cryo_temp != 'RT':
    #             fn = f'FS_{scan_subtype}_{cryo_temp}{scan_number:04d}'
    #         else:
    #             raise NotImplementedError
    #     else:
    #         raise ValueError(f'get_filename with ({cryo_temp, scan_type, scan_subtype, scan_number}) failed')
    #     return fn
    def _get_filename(light_source, scan_type, scan_number):
        if scan_type == 'deflector_scan':
            fn = f'OMBE_{light_source}_3D{scan_number:04d}'
        elif scan_type == 'phi_motor_scan':
                raise ValueError(f'phi motor scans require a filename input')
        else:
            raise ValueError(f'get_filename with ({light_source, scan_type, scan_number}) failed')
        return fn





if __name__ == '__main__':
    # d = Data2D.single_load('October', k_cut='GM', scan_number=1)
    # d2 = Data3D.single_load('October', scan_subtype='coarse', scan_number=1)
    # d2.berend_data.show(mode = 'CE', val = 15, Eint = 0.02)
    #
    # d = Data2D.single_load('October', cryo_temp='9K', k_cut='GM')
    # d = Data2D.single_load('October', scan_type='UPS', scan_number=1)
    # from AliPlottingFunctions import plot2D
    # plot2D(d.xaxis, d.yaxis, d)
    # d = Data3D.single_load('October', cryo_temp='140K', scan_subtype='Fermi_Surface')
    # d.berend_data.show(mode='CE', val=15, Eint=0.02)
    # #
    # d9K = Data3D.single_load('October', cryo_temp='9K', scan_subtype='Fermi_Edge')
    # d9K.berend_data.show(mode='CE', val=16, Eint=0.02)
    #
    from plotting_functions import plot2D, plot3D
    d = Data2D.single_load('January', scan_number=3)
    d = Data2D.single_load('January', light_source='XUV', scan_number=3)
    d = Data2D.single_load('January', light_source='XUV', filename='OMBE_XUV_2D_2.ibw')
    plot2D(d.xaxis, d.yaxis, d.data)

    d2 = Data3D.single_load('January', scan_type='deflector_scan', scan_number=1)
    d2.berend_dataclass.show(mode ='CE', val = 15, Eint = 0.02)
    plot3D(x=d2.yaxis, y=d2.xaxis, z=d2.zaxis, data=np.moveaxis(d2.data, 2, 1), slice_dim='z', slice_val=15,
           int_range=0.02)