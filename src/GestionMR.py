import FuentesClass as FuentesClass
import opt as opt
import pyomo.environ as pyo
from pyomo.core import value

import pandas as pd
import string
import random
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
plt.rcParams['text.usetex'] = True
import numpy as np

import json
import sys
import os
import time


def read_data(forecast_filepath, demand_filepath, sepr=','):#, filepath_bat):

    #Identify delimiter
    with open(forecast_filepath, newline='') as forecast:
        dialect = csv.Sniffer().sniff(forecast.read(1024))
    forecast_df = pd.read_csv(forecast_filepath, sep=dialect.delimiter, header=0, index_col='t')

    with open(demand_filepath, newline='') as forecast:
        dialect = csv.Sniffer().sniff(forecast.read(1024))
    demand = pd.read_csv(demand_filepath, squeeze=True, sep=dialect.delimiter, header=0)['demand'].to_dict()
    
    return forecast_df, demand

def create_generators(param_filepath, forecast_df):
    with open(param_filepath) as parameters:
        data = json.load(parameters)
    
    generators = data['generators']
    battery = data['battery']

    battery = FuentesClass.Bateria(*battery.values())

    generators_dict = {}
    for i in generators:
        if i['tec'] == 'S':
            obj_aux = FuentesClass.Solar(*i.values())
            obj_aux.generation(forecast_df=forecast_df)
        elif i['tec'] == 'W':
            obj_aux = FuentesClass.Eolica(*i.values())
            obj_aux.generation(forecast_df=forecast_df)
        elif i['tec'] == 'H':
            obj_aux = FuentesClass.Hidraulica(*i.values())
        elif i['tec'] == 'D':
            obj_aux = FuentesClass.Diesel(*i.values())
        elif i['tec'] == 'NA':
            obj_aux = FuentesClass.Fict(*i.values())
        # else:
        #     raise RuntimeError('Generator ({}) with unknow tecnology ({}).'.format(i['id_gen'], i['tec'])
        
        generators_dict[i['id_gen']] = obj_aux
    
        
    return generators_dict, battery

class Results():
    def __init__(self, model):
        #results = {}
        G_data = {'g_'+i: [0]*len(model.T) for i in model.calI}
        for (i,t), v in model.G.items():
            G_data['g_'+i][t] = value(v)
        
        self.g = pd.DataFrame(G_data, columns=[*G_data.keys()])

        x_data = {'x_'+i: [0]*len(model.T) for i in model.I}
        for (i,t), v in model.x.items():
            x_data['x_'+i][t] = value(v)
        
        self.x = pd.DataFrame(x_data, columns=[*x_data.keys()])

        b_data = [0] * len(model.T)
        
        for t, v in model.B.items():
            b_data[t] = value(v)
        b_data = {'b': b_data}
        
        self.b = pd.DataFrame(b_data)

        eb_data = [0]*len(model.T)
        for t in model.T:
            eb_data[t] = value(model.EB[t])
        
        self.eb = pd.DataFrame(eb_data)



    def export_results(self, opt_results, base_file_name):
        
        self.term_cond = opt_results.solver.termination_condition
        
        dt = self.g.copy()
        for i in self.x.columns:
            dt[i] = self.x[i]
        for i in self.b.columns:
            dt[i] = self.b[i]
        
        new_path = '/results/'+base_file_name
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        else:
            folder_name = base_file_name+'_'+str(random.choice(string.ascii_letters))+str(random.randint(0, 10))
            new_path = '/results/'+folder_name
            os.makedirs(new_path)
                
        
        #Creating the .CSV file
        dt.to_csv(new_path+'/'+base_file_name+'.csv')
        print('Your output file ({}) is in ({}) folder'.format(base_file_name, new_path))

        sys.exit()

        with open('../results/results.csv', 'a') as file:
            writer = csv.writer(file)
            writer.writerow([folder_name, down_limit, up_limit, l_min, l_max, 
                execution_time, pyo.value(model.generation_cost), term_cond])
        

        return folder_name

def visualize_results(g_df, x_df, d_df, eb_df, label=None, name=None):
    data_dem = list(d_df.values())
    
    data_s = g_df['Solar1'].to_numpy()
    data_w = g_df['Wind1'].to_numpy()
    data_d = g_df['Diesel1'].to_numpy()
    data_b = g_df['Battery1'].to_numpy()
    data_f = g_df['Fict'].to_numpy()
    data_eb = eb_df[0].to_numpy() * -1
    t = np.arange(len(data_s))
    
    
    

    plt.figure(figsize=(11,6))
    plt.ylim(min(data_eb)-50, max(data_w+data_s+data_b+data_d+data_f)+200)
    
    plt.bar(t,data_eb,color="black",label="Charge")
    plt.bar(t,data_w,color="limegreen",label="Wind")
    plt.bar(t,data_s,color="gold",bottom=data_w,label="Solar")
    plt.bar(t,data_d,color="saddlebrown",bottom=data_w+data_s,label="Diesel")
    plt.bar(t,data_b,color="royalblue",bottom=data_w+data_s+data_d,label="Battery")
    plt.bar(t,data_f,color="gray",bottom=data_w+data_s+data_d+data_b,label="Unattended demand")
    
    plt.plot(data_dem, color='red', marker='o', linestyle='dashed', linewidth=3, markersize=4, label="Demand")
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


if __name__ == "__main__":
    
    locations = ['P', 'SA', 'PN']
    days = ['01', '02', '03', '04', '05', '06', '07']
    
    # Just Test conditions
    location = 'ME'
    day = '01'
    forecast_filepath = '../data/instances/P/P03FORECAST.csv'
    demand_filepath = '../data/instances/P/P03DEMAND.csv'
    param_filepath = '../data/parameters_P.json'
    

    # forecast_filepath = os.path.join('../data/instances', str(location+day+'FORECAST.csv'))
    # demand_filepath = os.path.join('../data/instances', str(location+day+'DEMAND.csv'))
    
    forecast_df, demand = read_data(forecast_filepath, demand_filepath, sepr=';')
    
    generators_dict, battery = create_generators(param_filepath)

    down_limit, up_limit, l_min, l_max = 0.2, 0.85, 4, 4

    
    
    model = opt.make_model(generators_dict, forecast_df, battery, demand,
                            down_limit, up_limit, l_min, l_max)

    opt = pyo.SolverFactory('gurobi')

    timea = time.time()
    results = opt.solve(model)
    execution_time = time.time() - timea

    term_cond = results.solver.termination_condition
    if term_cond != pyo.TerminationCondition.optimal:
        print ("Termination condition={}".format(term_cond))
        raise RuntimeError("Optimization failed.")
    

    # G_df, x_df, b_df, eb_df = create_results(model)

    r = Results(model)
    r.export_results(results, "Prueba")
    """
    folder_name = export_results(model, location, day, x_df, G_df, b_df,
        execution_time, down_limit, up_limit, l_max, l_min, term_cond)
    
    print("Resultados en la carpeta: "+folder_name)
    #model.EL.pprint()
    #model.EB.pprint()
    #model.temp.pprint()
    """
    # visualize_results(G_df, x_df, b_df, eb_df)
    # model.G.pprint()
    
    
    