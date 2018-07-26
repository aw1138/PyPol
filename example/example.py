# Example code

## general & for pol
import numpy as np
import glob
import pandas as pd
from astropy.io import fits
from astropy import units as u
import itertools
from astropy.time import Time
import matplotlib.pyplot as plt

## for times
from astropy import time, coordinates as coord, units as u
from astropy.units import imperial
from astropy.units import cds

## main import
from PyPol import PyPol

#
redfils = {'filters':['R', 'I'], 'paths':['fils/R.fil', 'fils/I.fil']}
bluefils = {'filters':['U', 'B'], 'paths':['fils/UX.fil', 'fils/B.fil']}

fils = [redfils, bluefils]

test = PyPol('example/data/*r_*.fits', 'example/data/*b_*.fits')

# # without ism removal

# test_table = test.find_pol(fils, 18.59773, 'HPOL_Sys_Err_Aislynn.txt')

# print(test_table[:10])

# with ism removal

# get star candidates
ism = pd.read_excel('example/data/ism_stars.xlsx')
candi = ism.drop(columns=['notes', 'spec', 'HD', 'glat', 'glong']).dropna().drop(index=[20, 31])
candi = candi.drop(index=[33,34,35,36])
teststars = candi.drop(index=[1, 7])

ismstars = (teststars['POL'], teststars['PA'].values, teststars['Pol-e'])

test_ism = test.find_pol(fils, 18.59773, 'example/data/HPOL_Sys_Err_Aislynn.txt', ism=True, stars=ismstars)

print(test_ism[:10])

# # timer
# import timeit

# def wrapper(func, *args, **kwargs):
#     def wrapped():
#         return func(*args, **kwargs)
#     return wrapped

# wrapped = wrapper(test.find_pol, fils, 18.59773, 'HPOL_Sys_Err_Aislynn.txt', ism=True, stars=ismstars)

# timeit.timeit(wrapped, number=10)