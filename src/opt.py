from mimetypes import init
import pyomo.environ as pyo
import pandas as pd
import sys
import itertools
from FuentesClass import Bateria, Diesel, Fict
import time
import csv
import json
import FuentesClass as FuentesClass

class _MG_model():
    """
    A class to solve microgrid management model depending wich model is selected by user.
    
    Args:
        model_name (str) name of the model used to be created
        
    Attributes:
        model_name
    """
    def __init__(self, model_name):
        self.model_name = model_name
    
    #Function to be call from Subclasses (i.e. Deterministic and AAED)
    def _solve_model(self, solver):
        optimizer = pyo.SolverFactory(solver)
        timea = time.time()
        results = optimizer.solve(self.model)
        execution_time = time.time() - timea
        term_cond = results.solver.termination_condition
        if term_cond != pyo.TerminationCondition.optimal:
            print ("Termination condition={}".format(term_cond))
            raise RuntimeError("Optimization failed.")
        else:
            print("Your model has been successfully solved in {} seconds.".format(execution_time))
        return self.model



class Deterministic(_MG_model):
    def __init__(self, forecast_filepath, demand_filepath, param_filepath,
        down_limit=0.2, up_limit=0.95, l_min=4, l_max=4, model_name='Deterministic'):
        """
        A class to create Microgrid Management model using a Deterministic approach.
        
        Args:
            forecast_filepath (str) weather forecast .csv path.
            demand_filepath (str) demand forecast .csv path.
            param_filepath (str) generators and battery .json path.
            down_limit (float) minimun percentage of battery level to enter into deep-discharge. default: 0.2
            up_limit (float) maximun percentage of battery level to enter into  overcharge. default: 0.95
            l_min (int) maximin number of deep-discharge periods allowed into the optimization horizon. default: 4
            l_max (int) maximin number of overcharge periods allowed into the optimization horizon. default: 4
            
        Attributes:
            model (Pyomo Concrete Model)
        """

        forecast_df, demand = self._read_data(forecast_filepath, demand_filepath, sepr=';')
        generators_dict, battery = self._create_generators(param_filepath)
        self.model = self._make_model(generators_dict, forecast_df, battery, demand,
                                        down_limit, up_limit, l_min, l_max)
        super().__init__(model_name)

    def solve(self, solver='cbc'):
        return self._solve_model(solver)
    
    def _read_data(forecast_filepath, demand_filepath, sepr=','):
        """
        A method to read weather and demand forecast.
        
        Args:
            forecast_filepath (str) weather forecast .csv path.
            demand_filepath (str) demand forecast .csv path.            
        Return:
            forecast_df (DataFrame) Weather forecast DataFrame.
            demand (DataFrame) Demand forecast DataFrame.
        """
        #Identify delimiter
        with open(forecast_filepath, newline='') as forecast:
            dialect = csv.Sniffer().sniff(forecast.read(1024))
        forecast_df = pd.read_csv(forecast_filepath, sep=dialect.delimiter, header=0, index_col='t')

        with open(demand_filepath, newline='') as forecast:
            dialect = csv.Sniffer().sniff(forecast.read(1024))
        demand = pd.read_csv(demand_filepath, squeeze=True, sep=dialect.delimiter, header=0)['demand'].to_dict()
        
        return forecast_df, demand

    def _create_generators(param_filepath):
        """
        A method to create Generators and Battery objects.
        
        Args:
            param_filepath (str) generators and battery .json path.          
        Return:
            generators_dict (dict) Dictionary with Generators objects.
            battery (Bateria) Bateria object.
        """
        with open(param_filepath) as parameters:
            data = json.load(parameters)
        
        generators = data['generators']
        battery = data['battery']

        battery = FuentesClass.Bateria(*battery.values())

        generators_dict = {}
        for i in generators:
            if i['tec'] == 'S':
                obj_aux = FuentesClass.Solar(*i.values())
            elif i['tec'] == 'W':
                obj_aux = FuentesClass.Eolica(*i.values())
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

    def _make_model(self, generators_dict=None, forecast_df=None, battery=None, demand=None,
        down_limit=None, up_limit=None, l_min=None, l_max=None):
        """
        Crea el modelo.
        
        Args:
            I (list) generadores instalados en la microred.
            times (int) numero de periodos de tiempo sobre los cuales se va a ejecutar el modelo.
            param (array) parametros propios para los generadores I.
            
        Returns:
            model (Pyomo ConcreteModel)
        """

        model = pyo.ConcreteModel(name="Deterministic Microgrid Management")
        
        model.I = pyo.Set(initialize=[i for i in generators_dict.keys()])
        model.bat = pyo.Set(initialize=[battery.id_bat])
        model.calI = model.I | model.bat
        model.T = pyo.Set(initialize=[i for i in range(len(forecast_df))])

        model.D = pyo.Param(model.T, initialize=demand)

        model.P = pyo.Param(initialize=5)

        model.up_limit = pyo.Param(initialize=up_limit)
        model.down_limit = pyo.Param(initialize= down_limit)
        model.l_min = pyo.Param(initialize=l_min)
        model.l_max = pyo.Param(initialize=l_max)

        model.x = pyo.Var(model.I, model.T, within=pyo.Binary, initialize=0)

        model.G = pyo.Var(model.calI, model.T, within=pyo.NonNegativeReals, initialize=0)

        model.EB = pyo.Var(model.T, within=pyo.NonNegativeReals, initialize=0) #Power intended for charging the storage unit in time interval t.

        model.B = pyo.Var(model.T, within=pyo.NonNegativeReals) #Power level in battery unit over time period t.

        model.y = pyo.Var(model.T, within=pyo.Binary, initialize=0)

        model.Ic = pyo.Var(model.T, within=pyo.NonNegativeReals) #Numero de periodos de carga continua

        model.z = pyo.Var(model.T, within=pyo.NonNegativeReals)

        model.Sc = pyo.Var(within=pyo.NonNegativeReals)

        model.pmin = pyo.Var(model.T, within=pyo.Binary, initialize=0)

        model.pmax = pyo.Var(model.T, within=pyo.Binary, initialize=0)

        def B_rule(model, t):
            if t == 0:
                expr = battery.eb_zero * (1-battery.o)
                expr += model.EB[t] * battery.ef
                expr -= (model.G[battery.id_bat, t]/battery.ef_inv)
                return model.B[t] == expr
            else:
                expr = model.B[t-1] * (1-battery.o)
                expr += model.EB[t] * battery.ef
                expr -= (model.G[battery.id_bat, t]/battery.ef_inv)
                return model.B[t] == expr

        model.B_rule = pyo.Constraint(model.T, rule=B_rule)

        model.EL = pyo.Var(model.T, within=pyo.NonNegativeReals, initialize=0) #Power generated at t time from S and W technologies to satisfying the demand.
        
        model.EW = pyo.Var(model.T, within=pyo.NonNegativeReals, initialize=0) #Electric power discarded due to over generation in time interval t.

        def Bconstraint_rule(model, t):
            # for b in model.bat:
            #     return (0, model.G[b,t], 0)
            return model.B[t] <= battery.zb

        model.Bconstraint = pyo.Constraint(model.T, rule=Bconstraint_rule)
        
        def G_rule(model, i, t):
            gen = generators_dict[i]
            if gen.tec == 'S':
                return model.G[i,t] == gen.G_test * (forecast_df['Rt'][t]/gen.R_test) * model.x[i,t]
            if gen.tec == 'W':
                if forecast_df['Wt'][t] < gen.w_min:
                    return model.G[i,t] == 0
                elif forecast_df['Wt'][t] < gen.w_a:
                    # return model.G[i,t] == 0
                    return model.G[i,t] == ( (1/2) * gen.p * gen.s * (forecast_df['Wt'][t]**3) * gen.ef *gen.n* model.x[i,t])/1000
                elif forecast_df['Wt'][t] <= gen.w_max:
                    return model.G[i,t] == ( (1/2) * gen.p * gen.s * (gen.w_a**3) * gen.ef *gen.n* model.x[i,t])/1000
                else:
                    return model.G[i,t] == 0
            if gen.tec == 'H':
                #A Corregir
                return model.G[i,t] == gen.p * 9.8 * gen.ht * gen.ef * forecast_df['Qt'][t] * model.x[i,t]
            if gen.tec == 'D':
                return 0 <= model.G[i,t] - model.x[i,t]*gen.g_min
            if gen.tec == 'G':
                return model.G[i,t] <= model.x[i,t]*gen.g_max
            if gen.tec == 'NA':
                return pyo.Constraint.Skip
                
        
        model.G_rule = pyo.Constraint(model.I, model.T, rule=G_rule)

        def maxG_diesel_rule(model, i, t):
            gen = generators_dict[i]
            if gen.tec == 'D':
                return 0 <= model.x[i,t]*gen.g_max - model.G[i,t]
            else:
                return pyo.Constraint.Skip
            
        model.maxG_diesel_rule = pyo.Constraint(model.I, model.T, rule=maxG_diesel_rule)

        def Gconstraint_rule(model, t):
            ls = sum(model.G[i,t] for i in model.I if generators_dict[i].tec == 'S' or 
                        generators_dict[i].tec == 'W' or generators_dict[i].tec == 'H' or
                        generators_dict[i].tec == 'D')

            rs = model.EL[t] + model.EB[t] + model.EW[t]
            return ls == rs

        model.Gconstraint = pyo.Constraint(model.T, rule=Gconstraint_rule)
        
        def Dconstraint_rule(model, t):
            
            rs = model.EL[t]
            rs += sum(model.G[b,t] for b in model.bat)
            if 'Fict' in generators_dict.keys():
                rs += model.G['Fict', t]
            # rs += sum(model.G[i,t] for i in model.I if generators_dict[i].tec == 'H')
            # rs += sum(model.G[i,t] for i in model.I if generators_dict[i].tec == 'D')
            # rs -= model.temp[t] #Variable que guarda la energia desperdiciada en las fuentes H y D
            
            return model.D[t] == rs #(model.D[t], rs)

        model.Dconstraint = pyo.Constraint(model.T, rule=Dconstraint_rule)
        #sys.exit()

        
        #----------Nuevas restricciones control de baterías (SEGUNDA OPCIÓN)
        #Tasa máxima de carga y descarga
        def maxDescRate_rule(model, t):
            return model.G[battery.id_bat, t] <= battery.mdr

        model.maxDescRate_constraint = pyo.Constraint(model.T, rule=maxDescRate_rule)

        def maxChaRate_rule(model, t):
            return model.EB[t] <= battery.mcr

        model.maxChaRate_constraint = pyo.Constraint(model.T, rule=maxChaRate_rule)

        

        def DescargaConstraint_rule(model, t):
            return battery.zb * model.y[t] >= model.G[battery.id_bat, t]
        
        model.DescargaConstraint = pyo.Constraint(model.T, rule=DescargaConstraint_rule)

        def ndc_rule(model, t):
            return battery.epsilon * model.y[t] <= model.G[battery.id_bat, t]
        
        model.ndc = pyo.Constraint(model.T, rule=ndc_rule)

        def Sconstraint_rule(model):
            return model.Sc == sum(model.z[t] for t in model.T)

        model.Sconstraint = pyo.Constraint(rule=Sconstraint_rule)

        def aux1_rule(model, t):
            if t < 1:
                return pyo.Constraint.Skip
            else:
                return model.y[t]-model.y[t-1] <= model.z[t]
        
        model.aux1_constraint = pyo.Constraint(model.T, rule=aux1_rule)

        def maxS_rule(model):
            return model.Sc <= battery.M
        
        model.maxS = pyo.Constraint(rule=maxS_rule)

        #Descarga profunda

        def deepDesc1_rule(model, t):
            return (model.B[t]/battery.zb) + model.pmin[t] >= model.down_limit
        
        model.deepDesc1_constraint = pyo.Constraint(model.T, rule=deepDesc1_rule)

        def deepDesc2_rule(model, t):
            return (model.B[t]/battery.zb) - (1 - model.pmin[t]) <= model.down_limit
        
        model.deepDesc2_constraint = pyo.Constraint(model.T, rule=deepDesc2_rule)

        #Sobrecarga
        def overload1_rule(model, t):
            return (model.B[t]/battery.zb) - model.pmax[t] <= model.up_limit
        
        model.overload1_constraint = pyo.Constraint(model.T, rule=overload1_rule)

        def overload2_rule(model, t):
            return (model.B[t]/battery.zb) + (1 - model.pmax[t]) >= model.up_limit
        
        model.overload2_constraint = pyo.Constraint(model.T, rule=overload2_rule)

        # Numero de periodos máximos en los que se entra en descarga profunda

        def l_min_rule(model):
            return sum(model.pmin[t] for t in model.T) <= model.l_min
        model.l_min_constraint = pyo.Constraint(rule=l_min_rule)

        # Numero de periodos máximos en los que se entra en sobrecarga

        def l_max_rule(model):
            return sum(model.pmax[t] for t in model.T) <= model.l_max
        model.l_max_constraint = pyo.Constraint(rule=l_max_rule)
        
        
        #Funcion objetivo

        def obj_rule(model):

            return sum(sum(generators_dict[i].va_op * model.G[i,t] for t in model.T)for i in model.I) +0.1*sum(model.EW[t] for t in model.T)# Incluir temporal

        model.generation_cost = pyo.Objective(rule=obj_rule)

        return model

class AAED(_MG_model):
    def __init__(self, generators_dict, battery, demand_filepath, solar_filepath, wind_filepath, P, T, weight, model_name='Affine Arithmetic'):
        #def __init__(self, param_filepath, D, S, W, P, T, weight, model_name='Affine Arithmetic'):
        """
        A class to create Microgrid Management model using Affine Arithmetic Economic Dispatch approach.
        
        Args:
            param_filepath (str) generators and battery .json path.
            demand_filepath (str) Demand forecast data and deviations .csv filepath.
            solar_filepath (str) Solar generation forecast data and deviations .csv filepath.
            wind_filepath (str) Wind generation forecast data and deviations .csv filepath.
            P (int) Number of noise symbols for Demand, Solar and Wind generation.
            T (int) Number of time periods for the optimization process.
            weight (float) Weight for mean case objective function.
            
        Attributes:
            model_name
        """


        # generators_dict, battery = self._create_generators(param_filepath)
        D, S, W = self._read_data(demand_filepath, solar_filepath, wind_filepath)
        self.model = self._make_model(generators_dict, battery, D, S, W, P, T, weight)
        super().__init__(model_name)
    
    def _read_data(self, demand_filepath, solar_filepath, wind_filepath):
        D = pd.read_csv(demand_filepath)
        S = pd.read_csv(solar_filepath)
        W = pd.read_csv(wind_filepath)

        d = {}
        s = {}
        w = {}

        d['mean'] = D['mean']
        s['mean'] = S['mean']
        w['mean'] = W['mean']

        D.drop('mean', inplace=True, axis=1)
        S.drop('mean', inplace=True, axis=1)
        W.drop('mean', inplace=True, axis=1)
        
        aux_d = {}
        aux_s = {}
        aux_w = {}
        
        for i in range(1, len(D.columns)+1):
            for t in range(len(D)):
                aux_d[('D', i), t] = D[str(i)][t]
        d['dev'] = aux_d

        for i in range(1, len(S.columns)+1):
            for t in range(len(S)):
                aux_s[('S', i), t] = S[str(i)][t]
        s['dev'] = aux_s

        for i in range(1, len(W.columns)+1):
            for t in range(len(W)):
                aux_w[('W', i), t] = W[str(i)][t]
        w['dev'] = aux_w

        return d, s, w
    
    def _create_generators(param_filepath):
        """
        A method to create Generators and Battery objects.
        
        Args:
            param_filepath (str) generators and battery .json path.          
        Return:
            generators_dict (dict) Dictionary with Generators objects.
            battery (Bateria) Bateria object.
        """
        with open(param_filepath) as parameters:
            data = json.load(parameters)
        
        generators = data['generators']
        battery = data['battery']

        battery = FuentesClass.Bateria(*battery.values())

        generators_dict = {}
        for i in generators:

            if i['tec'] == 'D':
                obj_aux = FuentesClass.Diesel(*i.values())
            elif i['tec'] == 'NA':
                obj_aux = FuentesClass.Fict(*i.values())
            else:
                raise RuntimeError('Generator ({}) with unavailable tecnology ({}).'.format(i['id_gen'], i['tec']))

            generators_dict[i['id_gen']] = obj_aux
            
        return generators_dict, battery
    
    def solve(self, solver='cbc'):
        results = self._solve_model(solver)
        return results
    
    def _epsilons(self, actuals):
        epsilon = {}

        for f, h in self.model.H:
            for t in model.T:
                if f == 'D':
                    aux = (actuals[f, t]-self.model.D_0[t])/ self.model.D[f, h, t]
                    if aux < -1:
                        epsilon[f, h, t] = -1
                    elif aux > 1:
                        epsilon[f, h, t] = 1
                    else:
                        epsilon[f, h, t] = aux
                    
                elif f == 'S':
                    aux = (actuals[f, t]-self.model.S_0[t])/ self.model.S[f, h, t]
                    if aux < -1:
                        epsilon[f, h, t] = -1
                    elif aux > 1:
                        epsilon[f, h, t] = 1
                    else:
                        epsilon[f, h, t] = aux
                elif f == 'W':
                    aux = (actuals[f, t]-self.model.W_0[t])/ self.model.W[f, h, t]
                    if aux < -1:
                        epsilon[f, h, t] = -1
                    elif aux > 1:
                        epsilon[f, h, t] = 1
                    else:
                        epsilon[f, h, t] = aux
        return epsilon

    def dispatch(self, actuals):
        
        epsilon = self._epsilons(actuals)

        g = {}
        for i in self.model.CalI:
            for t in self.model.T:
                g[i, t] = self.model.g[i, 0, t] + sum(self.model.g[i, f, h, t]*epsilon[f, h, t] for f, h in self.model.H)

        b = {}
        for t in self.model.T:
            b[t] = self.model.b[0, t] + sum(self.model.b[f, h, t]*epsilon[f, h, t] for f, h in self.model.H)
        
        eb = {}
        for t in self.model.T:
            eb[t] = self.model.eb[0, t] + sum(self.model.eb[f, h, t]*epsilon[f, h, t] for f, h in self.model.H)
        
        return g, b, eb

    def _make_model(self, generators_dict, battery, D, S, W, P, T, weight):
        model = pyo.ConcreteModel(name="Stochastic Microgrid Management")

        #Microgrid components
        model.T = pyo.Set(initialize=[i for i in range(len(D['mean']))])
        model.I = pyo.Set(initialize=[i for i in generators_dict.keys()])
        model.bat = pyo.Set(initialize=[battery.id_bat])
        model.calI = model.I | model.bat
        
        #Affine aithmetic sets
        model.zero = pyo.Set(initialize=[0])
        model.Pd = pyo.Set(initialize=list(itertools.product(['D'],range(1,P['D'])))) #Demand noise symbols set
        model.Ps = pyo.Set(initialize=list(itertools.product(['S'],range(1,P['S'])))) #Solar generation noise symbols set
        model.Pw = pyo.Set(initialize=list(itertools.product(['W'],range(1,P['W'])))) #Wind generation noise symbols set
        
        model.calPd = model.zero | model.Pd
        model.calPs = model.zero | model.Ps
        model.calPw = model.zero | model.Pw
        
        model.H = model.Pd | model.Ps | model.Pw #(1, ..., pd+ps,pw)
        model.calH = model.zero | model.H # (0, ..., pd+ps+pw)
        
        
        va_op_dict = {}
        for i in model.I:
            va_op_dict[i] = generators_dict[i].va_op
        
        
        #Params
        model.w = pyo.Param(initialize=weight) #Multi-objective weight for mean case
        model.D_0 = pyo.Param(model.T, initialize=D['mean'])
        model.D = pyo.Param(model.H, model.T, initialize=D['dev'], default=0)
        model.S_0 = pyo.Param(model.T, initialize=S['mean'])
        model.S = pyo.Param(model.H, model.T, initialize=S['dev'], default=0)
        model.W_0 = pyo.Param(model.T, initialize=W['mean'])
        model.W = pyo.Param(model.H, model.T, initialize=W['dev'], default=0)
        model.bat_cap = 100
        model.va_op = pyo.Param(model.I, initialize = va_op_dict)
        
        """
        Variables
        """
        model.g = pyo.Var(model.calI, model.calH, model.T, within=pyo.Reals) #Diesel generation
        # model.g_s = pyo.Var(model.T, model.calH, within=pyo.Reals)
        # model.g_w = pyo.Var(model.T, model.calH, within=pyo.Reals)
        model.b = pyo.Var(model.calH, model.T, within=pyo.Reals)
        model.eb = pyo.Var(model.calH, model.T, within=pyo.Reals)
        model.y = pyo.Var(model.T, model.calH, within=pyo.Reals)
        
        """
        Constraints  
        """
        def PBC_0_rule(model, t):
            # Mean value
            return sum(model.g[i, 0, t] for i in model.calI) + model.S_0[t] + model.W_0[t] == model.D_0[t] + model.eb[0, t]            
        model.PBC_0 = pyo.Constraint(model.T, rule=PBC_0_rule)
        
        def PBC_rule(model, f, h, t):
            # Deviations
            return sum(model.g[i, f, h, t] for i in model.calI) + model.S[f, h, t] + model.W[f, h, t] == model.D[f, h, t] + model.eb[f, h, t] 
        model.PBC = pyo.Constraint(model.H, model.T, rule=PBC_rule)
        
        def Bconstraint_rule(model, t):
            # Battery max cap
            return model.b[0, t] + sum(abs(model.b[h, t]) for h in model.H) <= model.bat_cap
        model.Bconstraint = pyo.Constraint(model.T, rule=Bconstraint_rule)
        
        def maxDescRate_rule(model, t):
            # Max discharge rate of the battery system
            return model.g[battery.id_bat, 0, t] + sum(abs(model.g[battery.id_bat, h, t]) for h in model.H) <= battery.mdr
        model.maxDescRate = pyo.Constraint(model.T, rule=maxDescRate_rule)

        def maxChaRate_rule(model, t):
            # Max charge rate of the battery system
            return model.eb[0, t] + sum(abs(model.eb[h, t]) for h in model.H) <= battery.mdr
        model.maxChaRate = pyo.Constraint(model.T, rule=maxChaRate_rule)
        
        def Bstate_0_rule(model, t):
            if t == 0:
                expr = battery.eb_zero * (1 - battery.o)
                expr += (model.eb[0, t]*battery.ef)
                expr -= (model.g[battery.id_bat,0,t]/battery.ef_inv)
                return model.b[0, t+1] == expr
            elif t == T-1:
                return pyo.Constraint.Skip
            else:
                expr = model.b[0, t] * (1 - battery.o)
                expr += (model.eb[0, t]*battery.ef)
                expr -= (model.g[battery.id_bat,0,t]/battery.ef_inv)
                return model.b[0, t+1] == expr
        model.Bstate_0 = pyo.Constraint(model.T, rule=Bstate_0_rule)
        
        def Bstate_rule(model, t):
            if t == 0:
                expr = sum(abs((battery.eb_zero * (1 - battery.o)) + (model.eb[0, t]*battery.ef) - 
                            (model.g[battery.id_bat,0,t]/battery.ef_inv)) for h in model.H)
                return sum(model.b[h, t+1] for h in model.H) == expr
            elif t == T-1:
                return pyo.Constraint.Skip
            else:
                expr = sum(abs((model.b[h, t] * (1 - battery.o)) + (model.eb[0, t]*battery.ef) - 
                            (model.g[battery.id_bat,0,t]/battery.ef_inv)) for h in model.H)
                return model.b[0, t+1] == expr
        model.Bstate = pyo.Constraint(model.T, rule=Bstate_rule)
        
        def obj_rule(model):
            obj = model.w * sum(sum(generators_dict[i].va_op * model.g[i, 0, t] for i in model.I) for t in model.T)
            obj += (1-model.w) * sum( abs(sum(sum(model.va_op[i] * model.g[i, h, t] for i in model.I) for t in model.T)) for h in model.H)
            return obj
        model.obj = pyo.Objective(rule=obj_rule)

        
        return model
    
    


if __name__ == "__main__":
    demand_filepath = '../data/stch/holiday/demand.csv'
    solar_filepath = '../data/stch/holiday/solar.csv'
    wind_filepath = '../data/stch/holiday/wind.csv'
    D = {'mean':[3, 4], 'dev':{(('D', 1), 0): 0.12, (('D', 1), 1): 0.3}}
    S = {'mean':[1, 2], 'dev':{(('S', 1), 0): 0.13, (('S', 1), 1): 0.1}}
    W = {'mean':[1, 1.3], 'dev':{(('W', 1), 0): 0.14, (('W', 1), 1): 0.5}}
    weight = 0.9 #Multi-objective weight for mean case
    Diesel1 = Diesel(id_gen = 'Diesel1', tec='D', va_op =50, ef=0.25, g_min=2, g_max=2.5)
    Fict1 = Fict(id_gen='Fict1', tec='NA', va_op=300)
    generators_dict = {'Diesel1':Diesel1, 'Fict1':Fict1}
    battery = Bateria(id_bat='Battery', ef=0.95, o=0.05, ef_inv=0.95, eb_zero=200, zb=500, epsilon=0.05, M=100, mcr=300, mdr=300)
    P = {'D': 2, 'S': 2, 'W': 2} # Debe automatizarse al leer los datos
    T = 24 # Debe automatizarse al leer los datos
    model = AAED(generators_dict, battery, demand_filepath, solar_filepath, wind_filepath, P, T, weight)
