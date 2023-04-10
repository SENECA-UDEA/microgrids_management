#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 10:54:40 2022

@author: mateoespitiaibarra
"""

import pandas as pd

filepath = '../data/stch/data_base/wind.csv'

df = pd.read_csv(filepath)

solar = {}

for i in range(int(len(df)/24)):
    solar[i] = list(df['kw'][i*24:(i+1)*24])

solar = pd.DataFrame(solar)
solar2 = solar.T

solar2.to_csv('../data/stch/data_base/wind_py.csv', index=False)