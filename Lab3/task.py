import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


temp = ctrl.Antecedent(np.arange(0, 101, 1), 'temperature')
flow = ctrl.Antecedent(np.arange(0, 101, 1), 'flow')

hot_valve = ctrl.Consequent(np.arange(-90, 91, 1), 'hot_valve')
cold_valve = ctrl.Consequent(np.arange(-90, 91, 1), 'cold_valve')

temp['cold'] = fuzz.trimf(temp.universe, [0, 0, 25])
temp['cool'] = fuzz.trimf(temp.universe, [15, 30, 45])
temp['warm'] = fuzz.trimf(temp.universe, [35, 50, 65])
temp['not_hot'] = fuzz.trimf(temp.universe, [55, 70, 85])
temp['hot'] = fuzz.trimf(temp.universe, [75, 100, 100])

flow['weak'] = fuzz.trimf(flow.universe, [0, 0, 40])
flow['medium'] = fuzz.trimf(flow.universe, [20, 50, 80])
flow['strong'] = fuzz.trimf(flow.universe, [60, 100, 100])

angles = {
    'big_left': [-90, -90, -60],
    'med_left': [-70, -45, -20],
    'small_left': [-30, -10, 0],
    'neutral': [-5, 0, 5],
    'small_right': [0, 10, 30],
    'med_right': [20, 45, 70],
    'big_right': [60, 90, 90]
}

for name, val in angles.items():
    hot_valve[name] = fuzz.trimf(hot_valve.universe, val)
    cold_valve[name] = fuzz.trimf(cold_valve.universe, val)

rule1 = ctrl.Rule(temp['hot'] & flow['strong'], (hot_valve['med_left'], cold_valve['med_right']))
rule2 = ctrl.Rule(temp['hot'] & flow['medium'], cold_valve['med_right'])
rule3 = ctrl.Rule(temp['not_hot'] & flow['strong'], hot_valve['small_left'])
rule4 = ctrl.Rule(temp['not_hot'] & flow['weak'], (hot_valve['small_right'], cold_valve['small_right']))
rule5 = ctrl.Rule(temp['warm'] & flow['medium'], (hot_valve['neutral'], cold_valve['neutral']))
rule6 = ctrl.Rule(temp['cool'] & flow['strong'], (hot_valve['med_right'], cold_valve['med_left']))
rule7 = ctrl.Rule(temp['cool'] & flow['medium'], (hot_valve['med_right'], cold_valve['small_left']))
rule8 = ctrl.Rule(temp['cold'] & flow['weak'], hot_valve['big_right'])
rule9 = ctrl.Rule(temp['cold'] & flow['strong'], (hot_valve['med_left'], cold_valve['med_right']))
rule10 = ctrl.Rule(temp['warm'] & flow['strong'], (hot_valve['small_left'], cold_valve['small_left']))
rule11 = ctrl.Rule(temp['warm'] & flow['weak'], (hot_valve['small_right'], cold_valve['small_right']))

system = ctrl.ControlSystem([
    rule1, rule2, rule3, rule4, rule5,
    rule6, rule7, rule8, rule9, rule10, rule11
])
sim = ctrl.ControlSystemSimulation(system)

sim.input['temperature'] = 80
sim.input['flow'] = 70
sim.compute()

print("Кут гарячого крана:", sim.output['hot_valve'])
print("Кут холодного крана:", sim.output['cold_valve'])

hot_valve.view(sim=sim)
cold_valve.view(sim=sim)
