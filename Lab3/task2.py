import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

temp = ctrl.Antecedent(np.arange(-10, 41, 1), 'temperature')
dtemp = ctrl.Antecedent(np.arange(-5, 6, 1), 'd_temperature')

regulator = ctrl.Consequent(np.arange(-90, 91, 1), 'regulator')

temp['very_cold'] = fuzz.trimf(temp.universe, [-10, -10, 5])
temp['cold'] = fuzz.trimf(temp.universe, [0, 10, 20])
temp['normal'] = fuzz.trimf(temp.universe, [15, 22, 28])
temp['warm'] = fuzz.trimf(temp.universe, [25, 30, 35])
temp['very_warm'] = fuzz.trimf(temp.universe, [30, 40, 40])

dtemp['negative'] = fuzz.trimf(dtemp.universe, [-5, -5, 0])
dtemp['zero'] = fuzz.trimf(dtemp.universe, [-1, 0, 1])
dtemp['positive'] = fuzz.trimf(dtemp.universe, [0, 5, 5])

angles = {
    'big_left': [-90, -90, -60],
    'small_left': [-45, -20, 0],
    'off': [-5, 0, 5],
    'small_right': [0, 20, 45],
    'big_right': [60, 90, 90]
}

for name, val in angles.items():
    regulator[name] = fuzz.trimf(regulator.universe, val)

rule1 = ctrl.Rule(temp['very_warm'] & dtemp['positive'], regulator['big_left'])
rule2 = ctrl.Rule(temp['very_warm'] & dtemp['negative'], regulator['small_left'])
rule3 = ctrl.Rule(temp['warm'] & dtemp['positive'], regulator['big_left'])
rule4 = ctrl.Rule(temp['warm'] & dtemp['negative'], regulator['off'])
rule5 = ctrl.Rule(temp['very_cold'] & dtemp['negative'], regulator['big_right'])
rule6 = ctrl.Rule(temp['very_cold'] & dtemp['positive'], regulator['small_right'])
rule7 = ctrl.Rule(temp['cold'] & dtemp['negative'], regulator['big_right'])
rule8 = ctrl.Rule(temp['cold'] & dtemp['positive'], regulator['off'])
rule9 = ctrl.Rule(temp['very_warm'] & dtemp['zero'], regulator['big_left'])
rule10 = ctrl.Rule(temp['warm'] & dtemp['zero'], regulator['small_left'])
rule11 = ctrl.Rule(temp['very_cold'] & dtemp['zero'], regulator['big_right'])
rule12 = ctrl.Rule(temp['cold'] & dtemp['zero'], regulator['small_right'])
rule13 = ctrl.Rule(temp['normal'] & dtemp['positive'], regulator['small_left'])
rule14 = ctrl.Rule(temp['normal'] & dtemp['negative'], regulator['small_right'])
rule15 = ctrl.Rule(temp['normal'] & dtemp['zero'], regulator['off'])

system = ctrl.ControlSystem([
    rule1, rule2, rule3, rule4, rule5,
    rule6, rule7, rule8, rule9, rule10,
    rule11, rule12, rule13, rule14, rule15
])

sim = ctrl.ControlSystemSimulation(system)

sim.input['temperature'] = 25
sim.input['d_temperature'] = 0

sim.compute()

print("Кут регулятора:", sim.output['regulator'])

regulator.view(sim=sim)
