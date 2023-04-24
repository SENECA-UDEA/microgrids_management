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

@click.group()
def menu():
    pass

@menu.command()
@click.option('--weather_forecast', '-wf', default=None, type=str, help='Path of weather forecast data .csv file')
@click.option('--demand_forecast', '-df', default=None, type=str, help='Path of demand forecast data .csv file')
@click.option('--generation_units', '-gu', default=None, type=str, help='Path of generation units parameters .json file')
@click.option('--down_limit', '-dl', default=0.2, help='Energy level at battery to enter into deep descharge status; default = 0.2')
@click.option('--up_limit', '-ul', default=0.9, help='Energy level at battery to enter into overcharge status; default = 0.9')
@click.option('--l_min', '-l_min', default=2, help='Maximum number of periods in which deep discharge is allowed; default = 2')
@click.option('--l_max', '-l_max', default=2, help='Maximum number of periods in which overcharge is allowed; default = 2')
@click.option('--solver_name', '-sn', default='gurobi', help='Solver name to be use to solve the model; default = gurobi')
@click.option('--model_name', '-mn', default='Deterministic', help='Model name for Pyomo object; default = Deterministic')
@click.option('--plot_results', '-plt', default=False, type=bool, help='Plot generation results')
@click.option('--base_file_name', '-bfn', default=None, help='Base name for .csv output file')
def d(weather_forecast, demand_forecast, generation_units, down_limit, up_limit, l_min, l_max, solver_name, model_name, plot_results, base_file_name):
    """
    Runs the deterministic module of microgrid_management package.
    """
    return main_det_func(weather_forecast, demand_forecast, generation_units, down_limit, up_limit, l_min, l_max, solver_name, model_name, plot_results, base_file_name)


def input_det_check(weather_forecast, demand_forecast, generation_units):
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


def main_det_func(weather_forecast, demand_forecast, generation_units, down_limit, up_limit, l_min, l_max, solver_name, model_name, plot_results, base_file_name):

    input_det_check(weather_forecast, demand_forecast, generation_units)

    # forecast_df, demand = GestionMR.read_data(weather_forecast, demand_forecast, sepr=';')
    # generators_dict, battery = GestionMR.create_generators(generation_units, forecast_df)

    rep = GestionMR.run_deterministic(weather_forecast, demand_forecast, generation_units, down_limit, up_limit, l_min, l_max, solver_name, model_name)

    if plot_results:
        GestionMR.visualize_results(rep)
    
    if base_file_name:
        rep.export_d(base_file_name)
    else:
        print(rep)

    return None


#################################### Stochastic
@menu.command()
@click.option('--solar_forecast', '-sf', default=None, type=str, help='Path of solar generation forecast data .csv file')
@click.option('--wind_forecast', '-wf', default=None, type=str, help='Path of wind generation forecast data .csv file')
@click.option('--demand_forecast', '-df', default=None, type=str, help='Path of demand forecast data .csv file')
@click.option('--actuals_filepath', '-af', default=None, type=str, help='Path of most recent forecast (Demand, Solar, Wind) data .csv file')
@click.option('--generation_units', '-gu', default=None, type=str, help='Path of generation units parameters .json file')
@click.option('--main_weight', '-mw', default=0.8, type=float, help='Main objective function weigth; default = 0.8')
@click.option('--solver_name', '-sn', default='gurobi', help='Solver name to be use to solve the model; default = gurobi')
@click.option('--model_name', '-mn', default='Deterministic', help='Model name for Pyomo object; default = Deterministic')
@click.option('--plot_results', '-plt', default=False, type=bool, help='Plot generation results')
@click.option('--base_file_name', '-bfn', default=None, help='Base name for .csv output file')
def s(solar_forecast, wind_forecast, demand_forecast, generation_units, actuals_filepath, main_weight, solver_name, model_name, plot_results, base_file_name):
    """
    Runs the stochastic module of microgrid_management package.
    """
    return main_stch_func(solar_forecast, wind_forecast, demand_forecast, generation_units, actuals_filepath, main_weight, solver_name, model_name, plot_results, base_file_name)


def input_stch_check(solar_forecast, wind_forecast, demand_forecast, actuals_filepath, generation_units):
    if not solar_forecast:
        raise RuntimeError('You have to set a solar_forecast input file')
    elif not os.path.exists(solar_forecast):
        raise RuntimeError('Data file ({}) does not exist in your current folder'.format(solar_forecast))
    elif not actuals_filepath:
        raise RuntimeError('You have to set an actuals_filepath input file')
    elif not os.path.exists(actuals_filepath):
        raise RuntimeError('Data file ({}) does not exist in your current folder'.format(actuals_filepath))
    elif not demand_forecast:
        raise RuntimeError('You have to set a demand_forecast input file')
    elif not os.path.exists(demand_forecast):
        raise RuntimeError('Data file ({}) does not exist in your current folder'.format(demand_forecast))
    elif not generation_units:
        raise RuntimeError('You have to set a generation_units input file')
    elif not os.path.exists(generation_units):
        raise RuntimeError('Data file ({}) does not exist in your current folder'.format(generation_units))


def main_stch_func(solar_forecast, wind_forecast, demand_forecast, generation_units, actuals_filepath, main_weight, solver_name, model_name, plot_results, base_file_name):

    input_stch_check(solar_forecast, wind_forecast, demand_forecast, actuals_filepath, generation_units)

    # forecast_df, demand = GestionMR.read_data(weather_forecast, demand_forecast, sepr=';')
    # generators_dict, battery = GestionMR.create_generators(generation_units, forecast_df)

    rep = GestionMR.run_AAED(solar_forecast, wind_forecast, demand_forecast, generation_units, actuals_filepath, main_weight, solver_name, model_name)

    if plot_results:
        GestionMR.visualize_results(rep)
    
    if base_file_name:
        rep.export_d(base_file_name)
    else:
        print(rep)

    return None

if __name__ == "__main__":
    menu()


# python __main__.py deterministic -wf /data/expr/det/FORECAST.csv -df /data/expr/det/DEMAND.csv -gu /data/expr/parameters.json