import pyomo.environ as pyo
from pyomo.util.infeasible import log_infeasible_constraints
import matplotlib.pyplot as plt
import numpy as np
from opt import Deterministic, AAED

import pandas as pd
import sys
import os

class Report():
    def __init__(self) -> None:
        self.rep = {}
        self.aux_stch = None

    def make_report_s(self, g, s, w, b, eb):
        
        df = pd.DataFrame()

        for i in g.keys():
            for t in g[i].keys():
                df[i] = g[i].values()

        df['s'] = s.values()

        df['w'] = w.values()

        df['b'] = b.values()

        df['eb'] = eb.values()

        self.rep = df.copy()
    
    def make_report_d(self, g, s, w, b, eb, x):
        df = pd.DataFrame()


        for i in g.keys():
            df[i] = g[i]
        
        df['s'] = s.values()

        df['w'] = w.values()

        df['b'] = b.values()

        df['eb'] = eb.values()

        self.rep = df.copy()
    
    def export_d(self, path):

        folder_name = path
        new_path = os.getcwd()+'/'+folder_name
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        self.rep.to_csv(new_path+'/{}.csv'.format(path))
        print('Your output file is located in {}'.format(new_path))
        

        

    
def visualize_results(g_df, x_df=None, d_df=None, label=None, name=None):
    #data_dem = list(d_df.values())
    
    data_s = g_df['s'].to_numpy()
    data_w = g_df['w'].to_numpy()
    data_d = g_df['Diesel1'].to_numpy()
    data_b = g_df['Battery1'].to_numpy()
    data_f = g_df['Fict'].to_numpy()
    data_eb = g_df['eb'].to_numpy() * -1
    t = np.arange(len(data_s))
    
    
    

    plt.figure(figsize=(11,6))
    plt.ylim(min(data_eb)-50, max(data_w+data_s+data_b+data_d+data_f)+200)
    
    plt.bar(t,data_eb,color="black",label="Charge")
    plt.bar(t,data_w,color="limegreen",label="Wind")
    plt.bar(t,data_s,color="gold",bottom=data_w,label="Solar")
    plt.bar(t,data_d,color="saddlebrown",bottom=data_w+data_s,label="Diesel")
    plt.bar(t,data_b,color="royalblue",bottom=data_w+data_s+data_d,label="Battery")
    plt.bar(t,data_f,color="gray",bottom=data_w+data_s+data_d+data_b,label="Unattended demand")
    
    #plt.plot(data_dem, color='red', marker='o', linestyle='dashed', linewidth=3, markersize=4, label="Demand")
    #plt.bar(t,data_eb,color="black",bottom=data_s+data_w+data_d+data_b+data_f,label="Charge")

    plt.xlabel("Time (hour)", fontsize=15)
    plt.ylabel("Generation (W)", fontsize=15)

    plt.legend(loc="best", ncol=len(g_df.columns)+2, fontsize=10.5)

    #loc = [0.5, 1]
    if label:
        plt.suptitle(label, fontsize=20)
    
    if name:
        plt.savefig(name+'.pdf', bbox_inches='tight')
    
    plt.show()
    
    return None
        

def run_deterministic(forecast_filepath, demand_filepath, param_filepath, 
                      down_limit=0.2, up_limit=0.95, l_min=4, l_max=4, solver_name='gurobi', model_name='Deterministic'):
    """
    This function runs the entire process to manage a microgrid usig Deterministic Optimization.
    In this, a 24 hours forecast is usted in order to optimize the Economic Dispatch variables.
    
    Args:
            forecast_filepath (str) weather forecast .csv path.
            demand_filepath (str) demand forecast .csv path.
            param_filepath (str) generators and battery .json path.
            down_limit (float) minimun percentage of battery level to enter into deep-discharge. default: 0.2
            up_limit (float) maximun percentage of battery level to enter into  overcharge. default: 0.95
            l_min (int) maximin number of deep-discharge periods allowed into the optimization horizon. default: 4
            l_max (int) maximin number of overcharge periods allowed into the optimization horizon. default: 4
    """
    
    report = Report()
    # exp_path = '../../data/expr/det/'

    # if not forecast_filepath:
    #     forecast_filepath = exp_path+'FORECAST.csv'
    # if not demand_filepath:
    #     demand_filepath = exp_path+'DEMAND.csv'
    
    # Create the Pyomo model
    model = Deterministic(forecast_filepath, demand_filepath, param_filepath,
                          down_limit, up_limit, l_min, l_max, model_name)
    
    model.solve(solver=solver_name)

    # model.model.G.pprint()
    
    g, s, w, b, eb, x = model.dispatch()

    report.make_report_d(g, s, w, b, eb, x)

    # print('Model {} - Objective Function: {}'.format(model.name, pyo.value(model.model.generation_cost)))

    # d = {t: pyo.value(v) for t, v in model.model.D.items()}
    
    # visualize_results(report.rep, label=model_name)

    return report.rep


def run_AAED(solar_filepath, wind_filepath, demand_filepath, param_filepath, actuals_filepath, weight, solver_name, model_name):
    """
    This function runs the entire process to manage a microgrid usig Affine Aritmetic Optimization.
    In this, a 24 hours forecast is usted in order to generate the AA coefficients for ED variables.
    After that, noise symbols are re-calculated each 4 hours in order to generate real dispatch
    values.
    This process needs 4 (24h/6h) "most recent information" forecast to re-calculate 
    noise symbols.
    """
    # exp_path = '../../data/expr/stch/'
    model = AAED(param_filepath, demand_filepath, solar_filepath, wind_filepath, weight, model_name)

    model.solve(solver=solver_name)

    g, s, w, b, eb = model.dispatch(actuals_filepath)

    report = Report()

    # for h in range(0, 24, 6):
    #     demand_forecast = exp_path+'forecast/{}_demand.csv'.format(h-6)
    #     solar_forecast = exp_path+'forecast/{}_solar.csv'.format(h-6)
    #     wind_forecast = exp_path+'forecast/{}_wind.csv'.format(h-6)
    #     print(h)
        
    #     model = AAED(param_filepath, demand_forecast, solar_forecast, wind_forecast, P, weight)
        
        
        
    #     model.solve(solver='gurobi')

        
        

    #     g, s, w, b, eb = model.dispatch(exp_path+'actuals/{}.csv'.format(h), h) # Return g, s, w, b, eb as dicts

        

    report.make_report_s(g, s, w, b, eb)
    
    # dd = pd.DataFrame()

    # for h in range(0, 24, 6):
    #     if h == 0:
    #         dd = report.rep[0].loc[:5].copy()
    #     else:
    #         tmp = report.rep[h].loc[:5].copy()
    #         #print(tmp)
    #         dd = pd.concat([dd, tmp])
            

    # Recalculate the value of the Objective Function according to the final load profile

    # of = sum(sum(z for z in pyo.value(model.model.va_op[i])*dd[i]) for i in model.model.I)
    # print('AFFINE OF: {}'.format(of))
    
    # d = {t: pyo.value(v) for t, v in model.model.D_0.items()}

    # visualize_results(dd, label='AAED')

    return report.rep


if __name__ == "__main__":
    
    locations = ['P', 'SA', 'PN']
    days = ['01', '02', '03', '04', '05', '06', '07']
    
    # Just Test conditions
    location = 'ME'
    day = '01'
    forecast_filepath = '../data/instances/P/P01FORECAST.csv'
    demand_filepath = '../data/instances/P/P03DEMAND.csv'
    param_filepath = '../data/instances/P/parameters_P.json'
    

    # forecast_filepath = os.path.join('../data/instances', str(location+day+'FORECAST.csv'))
    # demand_filepath = os.path.join('../data/instances', str(location+day+'DEMAND.csv'))
    
    forecast_df, demand = read_data(forecast_filepath, demand_filepath, sepr=';')
    
    generators_dict, battery = create_generators(param_filepath)

    down_limit, up_limit, l_min, l_max = 0.2, 0.85, 4, 4

    
    model = opt.Deterministic(generators_dict, forecast_df, battery, demand, down_limit, up_limit, l_min, l_max)
    results = model.solve('gurobi')
    # model = opt.make_model(generators_dict, forecast_df, battery, demand,
    #                         down_limit, up_limit, l_min, l_max)

    # optimizer = pyo.SolverFactory('gurobi')

    # timea = time.time()
    # results = optimizer.solve(model)
    # execution_time = time.time() - timea

    # term_cond = results.solver.termination_condition
    # if term_cond != pyo.TerminationCondition.optimal:
    #     print ("Termination condition={}".format(term_cond))
    #     raise RuntimeError("Optimization failed.")
    

    # G_df, x_df, b_df, eb_df = create_results(model)
    # """
    # folder_name = export_results(model, location, day, x_df, G_df, b_df,
    #     execution_time, down_limit, up_limit, l_max, l_min, term_cond)
    
    # print("Resultados en la carpeta: "+folder_name)
    # #model.EL.pprint()
    # #model.EB.pprint()
    # #model.temp.pprint()
    # """
    # visualize_results(G_df, x_df, demand, eb_df)
    # model.G.pprint()
    
    
    