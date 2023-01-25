# -*- coding: utf-8 -*-
"""Codigo dataSENECA.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_xw2GN-6XCR95iIyPKjuPO0sEWnf94bM
"""

import pandas as pd 
import numpy as np
import statistics as st
import datetime 
import calendar
import holidays


def read_data(path):
    
    co_holidays= holidays.Colombia() # Se buscan los festivos del año
    df = pd.read_csv(path,sep=",") # Se lee la data
    datad=df.copy()
    
    x=[datetime.date(2019,1,1)+datetime.timedelta(i) for i in range(0,365)] #Se generan las fechas
    datad['fecha']=pd.to_datetime(x)
    
    datad['dsemana']=datad.fecha.dt.day_name()
    
    
    festivo=[] #Lista boleana de festivos
    for i in datad.index:
      festivo.append(datad.fecha[i].date() in co_holidays)
    datad['dfestivo']=festivo
    
    
    datad['dfestivo']=datad['dfestivo'].astype(str)
    # Se escogen los dias de semana y se almacenan en "diasemana"
    diasemana=(datad.query("dfestivo=='False' and dsemana=='Monday' | dfestivo=='False' and dsemana=='Tuesday' | dfestivo=='False' and dsemana== 'Wednesday' | dfestivo=='False' and dsemana=='Thursday' | dfestivo=='False' and dsemana=='Friday'"))
    
    # Se escogen los sabados y se almacenan en "sabado"
    sabado=datad[(datad.dsemana=='Saturday') & (datad.dfestivo=='False')]
    
    # Se escogen los festivos y se almacenan en "holiday"
    holiday=(datad[(datad.dfestivo=='True') | (datad.dsemana=='Sunday')])
    
    #Se borran las columnas que no se necesitan
    diasemana=diasemana.drop(['dsemana', 'dfestivo','fecha'], axis=1)
    sabado=sabado.drop(['dsemana', 'dfestivo','fecha'], axis=1)
    holiday=holiday.drop(['dsemana', 'dfestivo','fecha'], axis=1)
    
    try:
        diasemana.drop('dias', inplace=True, axis=1)
        sabado.drop('dias', inplace=True, axis=1)
        holiday.drop('dias', inplace=True, axis=1)
    except:
        pass
    
    weekdesv=diasemana.std()
    satdesv=sabado.std()
    holiddesv=holiday.std()
    weekmean=diasemana.mean()
    satmean=sabado.mean()
    holidmean=holiday.mean()
    
    
    week={'mean': weekmean,1: weekdesv}
    sat={'mean': satmean,1: satdesv}
    holid={'mean': holidmean,1: holiddesv}
    
    
    
    week = pd.DataFrame(week)
    sat = pd.DataFrame(sat)
    holid = pd.DataFrame(holid)
    
    # week['mean'].plot.line(color='blue', label='Week')
    # sat['mean'].plot.line(color='green', label='Saturday')
    # holid['mean'].plot.line(color='red', label='Holiday')
    
    tbp = {'Weekday': week['mean'], 'Saturday': sat['mean'], 'Holiday': holid['mean']}
    tbp = pd.DataFrame(tbp)
    tbp.plot.line()
    
    return week, sat, holid
    
    """# Nueva sección"""
    
if __name__ == "__main__":

    week, sat, holid = read_data("data_base/wind_py.csv")
    week.to_csv('week/wind.csv', index=False)
    sat.to_csv('saturday/wind.csv', index=False)
    holid.to_csv('holiday/wind.csv', index=False)




