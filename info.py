
import pandas as pd
import numpy as np
import json

objective_prox = np.zeros((500, 4))
objective_ranking = np.zeros((500, 4))
objective_random = np.zeros((500, 4))

df_objective = pd.DataFrame()

with open(f"ranking.json", 'r') as file:
    sol1 = json.load(file)

with open(f"random.json", 'r') as file:
    sol2 = json.load(file)

    penalty = sol1["sol 1"]
    ranking = sol1["sol 2"]
    random = sol2["sol 2"]

    for h in range(500):
        if ranking[h]["Objective"] > 1000:
            ranking[h]["Objective"] -= 1000

        if random[h]["Objective"] > 1000:
            random[h]["Objective"] -= 1000

        objective_prox[h, 0] = penalty[h]["Objective"]
        objective_ranking[h, 0] = ranking[h]["Objective"]
        objective_random[h, 0] = random[h]["Objective"]

        objective_prox[h, 1] = penalty[h]["Caidos en Stockot"]
        objective_ranking[h, 1] = ranking[h]["Caidos en Stockot"]
        objective_random[h, 1] = random[h]["Caidos en Stockot"]

        objective_prox[h, 2] = penalty[h]["Horas en Stockout"]
        objective_ranking[h, 2] = ranking[h]["Horas en Stockout"]
        objective_random[h, 2] = random[h]["Horas en Stockout"]

        objective_prox[h, 3] = objective_prox[h, 0]/1000 + objective_prox[h, 1] + 0.125*objective_prox[h, 2]
        objective_ranking[h, 3] = objective_ranking[h, 0]/1000 + objective_ranking[h, 1] + 0.125*objective_ranking[h, 2]
        objective_random[h, 3] = objective_random[h, 0]/1000 + objective_random[h, 1] + 0.125*objective_random[h, 2]

        df_objective["Proximidad Transporte"] = objective_prox[:, 0]
        df_objective["Proximidad Fijo Stockout"] = objective_prox[:, 1]
        df_objective["Proximidad Variable Stockout"] = objective_prox[:, 2]
        df_objective["Proximidad Costo Total"] = objective_prox[:, 3]
        df_objective["Ranking Transporte"] = objective_ranking[:, 0]
        df_objective["Ranking Fijo Stockout"] = objective_ranking[:, 1]
        df_objective["Ranking Variable Stockout"] = objective_ranking[:, 2]
        df_objective["Ranking Costo Total"] = objective_ranking[:, 3]
        df_objective["Rand Transporte"] = objective_random[:, 0]
        df_objective["Rand Fijo Stockout"] = objective_random[:, 1]
        df_objective["Rand Variable Stockout"] = objective_random[:, 2]
        df_objective["Rand Costo Total"] = objective_random[:, 3]

writer = pd.ExcelWriter('objetivo_final.xlsx', engine='xlsxwriter')

df_objective.to_excel(writer, sheet_name='Objective')

writer.save()