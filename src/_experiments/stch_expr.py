from opt import AAED, Deterministic
import pyomo.environ as pyo
from pyomo.util.infeasible import log_infeasible_constraints


import pandas as pd
import sys

class Report():
    def __init__(self) -> None:
        self.rep = {}

    def make_report_s(self, g, s, w, b, eb, h):
        
        df = pd.DataFrame()

        for i in g.keys():
            for t in g[i].keys():
                df[i] = g[i].values()

        df['s'] = s.values()

        df['w'] = w.values()

        df['b'] = b.values()

        df['eb'] = eb.values()

        self.rep[h] = df
    
    def make_report_d(self, g, x, b, eb):
        df = pd.DataFrame()


        for i in g.keys():
            df[i] = g[i]
        

        df['b'] = b.values()

        df['eb'] = eb.values()

        self.rep = df.copy()
        

def run_deterministic(param_filepath):
    """
    This function runs the entire process to manage a microgrid usig Deterministic Optimization.
    In this, a 24 hours forecast is usted in order to optimize the Economic Dispatch variables.
    """
    
    report = Report()
    exp_path = '../../data/expr/det/'

    forecast_filepath = exp_path+'FORECAST.csv'
    demand_filepath = exp_path+'DEMAND.csv'
    
    # Create the Pyomo model
    model = Deterministic(forecast_filepath, demand_filepath, param_filepath)
    
    model.solve(solver='gurobi')
    
    g, x, b, eb = model.dispatch()

    report.make_report_d(g, x, b, eb)

    return report.rep


def run_AAED(param_filepath, demand_filepath, solar_filepath, wind_filepath, P, T, weight):
    """
    This function runs the entire process to manage a microgrid usig Affine Aritmetic Optimization.
    In this, a 24 hours forecast is usted in order to generate the AA coefficients for ED variables.
    After that, noise symbols are re-calculated each 4 hours in order to generate real dispatch
    values.
    This process needs 4 (24h/6h) "most recent information" forecast to re-calculate 
    noise symbols.
    """
    exp_path = '../../data/expr/stch/'
    # model = AAED(param_filepath, demand_filepath, solar_filepath, wind_filepath, P, T, weight)

    # model.solve(solver='gurobi')

    # g, b, eb = model.dispatch(exp_path+'actuals/')

    report = Report()

    for h in range(0, 24, 6):
        demand_forecast = exp_path+'forecast/{}_demand.csv'.format(h-6)
        solar_forecast = exp_path+'forecast/{}_solar.csv'.format(h-6)
        wind_forecast = exp_path+'forecast/{}_wind.csv'.format(h-6)
        print(h)
        
        model = AAED(param_filepath, demand_forecast, solar_forecast, wind_forecast, P, weight)
        
        
        
        model.solve(solver='gurobi')

        
        

        g, s, w, b, eb = model.dispatch(exp_path+'actuals/{}.csv'.format(h), h) # Return g, s, w, b, eb as dicts

        

        report.make_report_s(g, s, w, b, eb, h)
    
    dd = pd.DataFrame()

    for h in range(0, 24, 6):
        if h == 0:
            dd = report.rep[0].loc[:5].copy()
        else:
            tmp = report.rep[h].loc[:5].copy()
            #print(tmp)
            dd = pd.concat([dd, tmp])
            

        
        

    return report.rep, dd

if __name__ == '__main__':
    demand_filepath = '../../data/expr/stch/base_demand.csv' #With base that means all positive deviations
    solar_filepath = '../../data/expr/stch/base_wind.csv'
    wind_filepath = '../../data/expr/stch/base_solar.csv'
    param_filepath = '../../data/expr/parameters.json'
    weight = 0.9 #Multi-objective weight for mean case
    P = {'D': 2, 'S': 2, 'W': 2} # Debe automatizarse al leer los datos
    T = 24 # Debe automatizarse al leer los datos

    rep, cons = run_AAED(param_filepath, demand_filepath, solar_filepath, wind_filepath, P, T, weight)
    
    mod = run_deterministic(param_filepath)
    # g = rep.model.g.get_values()
    # eb = rep.model.eb.get_values()
    # for t in rep.model.T:
    #     s = (g['Battery1', 0, t] + g['Battery1','S', 1, t] +
    #          g['Battery1', 'D',1, t]+ g['Battery1', 'W',1, t])
    #     print(s)
    
    
    
    