import src.FuentesClass as FuentesClass
import src.opt as opt
import src.GestionMR as GestionMR
import pandas as pd
import json
import sys
import time
import pyomo.environ as pyo
from pyomo.core import value



import os
import click

@click.command()
@click.option('--weather_forecast', '-wf', default=None, type=str, help='Path of weather forecast data .csv file')
@click.option('--demand_forecast', '-df', default=None, type=str, help='Path of demand forecast data .csv file')
@click.option('--generation_units', '-gu', default=None, type=str, help='Path of generation units parameters .json file')
@click.option('--down_limit', '-dl', default=0.2, help='Energy level at battery to enter into deep descharge status; default = 0.2')
@click.option('--up_limit', '-ul', default=0.9, help='Energy level at battery to enter into overcharge status; default = 0.9')
@click.option('--l_min', '-l_min', default=2, help='Maximum number of periods in which deep discharge is allowed; default = 2')
@click.option('--l_max', '-l_max', default=2, help='Maximum number of periods in which overcharge is allowed; default = 2')
@click.option('--solver_name', '-sn', default='gurobi', help='Solver name to be use to solve the model; default = gurobi')
@click.option('--plot_results', '-plt', default=False, type=bool, help='Plot generation results')
@click.option('--base_file_name', '-bfn', default=None, help='Base name for .csv output file')
def main(weather_forecast, demand_forecast, generation_units, down_limit, up_limit, l_min, l_max, solver_name, plot_results, base_file_name):
    return main_func(weather_forecast, demand_forecast, generation_units, down_limit, up_limit, l_min, l_max, solver_name, plot_results, base_file_name)


def input_check(weather_forecast, demand_forecast, generation_units):
    if not weather_forecast:
        raise RuntimeError('You have to set a weather_forecast input file')
    elif not os.path.exists(weather_forecast):
        raise RuntimeError('Data file ({}) does not exist in your current folder'.format(weather_forecast))
    elif not demand_forecast:
        raise RuntimeError('You have to set a demand_forecast input file')
    elif not os.path.exists(demand_forecast):
        raise RuntimeError('Data file ({}) does not exist in your current folder'.format(demand_forecast))
    elif not generation_units:
        raise RuntimeError('You have to set a generation_units input file')
    elif not os.path.exists(generation_units):
        raise RuntimeError('Data file ({}) does not exist in your current folder'.format(generation_units))


def main_func(weather_forecast, demand_forecast, generation_units, down_limit, up_limit, l_min, l_max, solver_name, plot_results, base_file_name):

    input_check(weather_forecast, demand_forecast, generation_units)

    forecast_df, demand = GestionMR.read_data(weather_forecast, demand_forecast, sepr=';')
    generators_dict, battery = GestionMR.create_generators(generation_units, forecast_df)

    model = opt.make_model(generators_dict, forecast_df, battery, demand, down_limit, up_limit, l_min, l_max)

    optimizer = pyo.SolverFactory(solver_name)

    timea = time.time()
    opt_results = optimizer.solve(model)
    execution_time = time.time() - timea

    if plot_results:
        raise RuntimeError("This feature is still under development.")

    term_cond = opt_results.solver.termination_condition
    if term_cond != pyo.TerminationCondition.optimal:
        print ("Termination condition={}".format(term_cond))
        raise RuntimeError("Optimization failed.")

    model_results = GestionMR.Results(model)
    
    if base_file_name:
        model_results.export_results(opt_results, base_file_name)
    else:
        return model_results



if __name__ == "__main__":
    main()