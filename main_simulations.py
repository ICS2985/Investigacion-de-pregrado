from ast import Pass
from secrets import choice
from unicodedata import category
import pandas as pd
import numpy as np
from numpy import NaN, indices
# from pyrsistent import T
import estimacion_stockout
import main_stockout as stockout_approach
import main_proximidad as proximity_approach
import json
from itertools import chain, combinations, permutations
import random



def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def point_reader(point):
    point = point[6:].lstrip('(').rstrip(')').split()
    point[0] = float(point[0])
    point[1] = float(point[1])

    return point


def point_generator(georeferences):
    number_georeferences = list()
    for point in georeferences:
        number_georeferences.append(point_reader(point))

    return number_georeferences


def info_generator(df, c1, c2):
    np.random.seed(2985)
    largo = len(df)
    df["Demanda"] = np.random.random_sample(largo)*c1
    df["Cantidad"] = np.random.random_sample(largo)*c2


def sort_random(data):
    estimacion = list(set(data["Estimacion"]))
    estimados = list(data["Estimacion"])
    cantidades = dict()
    for x in estimacion:
        cantidades[x] = list(data["Estimacion"]).count(x)
    probs = []
    cont = len(estimacion)
    elegidos = []
    indices = []
    for i in estimacion:
        probs += [i]*cont
        cont -= 1
    while probs:
        elegido = random.choice(probs)
        if elegidos.count(elegido) < cantidades[elegido]:
            indice = estimados.index(elegido)
            estimados[indice] = -1
            elegidos.append(elegido)
            indices.append(indice)
        else:
            probs = list(filter(lambda a: a != elegido, probs))
    return indices


def read_data(combinacion, orden):
    random.seed(123)
    file = pd.read_csv("locations.csv")
    file.dropna(axis=0, inplace=True)
    locations = file[file["City"] == "Rochester"]
    georeferences = point_generator(list(locations["Georeference"]))
    atm_information = pd.DataFrame(columns=["Ubicacion"])
    atm_information["Ubicacion"] = georeferences
    demanda, capacidad_maxima = (5, 20)
    info_generator(atm_information, demanda, capacidad_maxima)

    nueva_fila = pd.DataFrame(data={'Ubicacion': [[-77.6486, 43.1496]], 'Demanda': NaN, 'Cantidad': NaN, 'Estimacion': 0, 'Turno': 0, 'Gap': 0})
    atm_information = nueva_fila.append(atm_information, ignore_index=True)
    estimacion_stockout.estimator(atm_information)
    atm_information["Tiempo Stockout"] = np.zeros(len(atm_information))
    atm_information["Tiempo Stockout Total"] = np.zeros(len(atm_information))
    atm_information["Caida stockout"] = np.zeros(len(atm_information))
    atm_information["Visitado"] = np.zeros(len(atm_information))
    atm_information["Tiempo Stockout (horas)"] = np.zeros(len(atm_information))
    atm_information["Visitado"].iloc[0] = NaN
    atm_information["Tiempo Stockout"].iloc[0] = NaN
    atm_information["Caida stockout"].iloc[0] = NaN
    atm_information["Tiempo Stockout Total"].iloc[0] = NaN
    atm_information["Tiempo Stockout (horas)"].iloc[0] = NaN
    atm_information.sort_values(by=combinacion, inplace=True, ascending=orden, na_position="first")

    """
    indices = sort_random(atm_information)
    ind = indices.index(0)
    indices.pop(ind)
    atm_information = atm_information.iloc[[0]+indices]
    """
    atm_information.reset_index(inplace=True, drop=True)

    return atm_information


def data_model(data_1, data_2, sol1, sol2, combinacion, orden):
    cantidad_1 = np.array(data_1["Cantidad"] - (1/3)*data_1["Demanda"])
    cantidad_2 = np.array(data_2["Cantidad"] - (1/3)*data_2["Demanda"])
    sol1["Horas en Stockout"] = 0
    sol2["Horas en Stockout"] = 0
    for i in range(5):
        for ruta in sol1[f"Vehicle {i}"]["Route"]:
            if data_1["Gap"].iloc[ruta[0]] > 0:
                cantidad_1[ruta[0]] = data_1["Gap"].iloc[ruta[0]]
                if max(ruta[2]-data_1["Hora"].iloc[ruta[0]], 0) > 0 and data_1["Estimacion"].iloc[ruta[0]] == 0 and data_1["Turno"].iloc[ruta[0]] == 0:
                    data_1["Tiempo Stockout (horas)"].iloc[ruta[0]] += max(ruta[2]-data_1["Hora"].iloc[ruta[0]], 0)
                    sol1["Horas en Stockout"] += max(ruta[2]-data_1["Hora"].iloc[ruta[0]], 0)
                data_1["Tiempo Stockout"].iloc[ruta[0]] = 0
            data_1["Visitado"].iloc[ruta[0]] += 1

        for ruta in sol2[f"Vehicle {i}"]["Route"]:
            if data_2["Gap"].iloc[ruta[0]] > 0:
                cantidad_2[ruta[0]] = data_2["Gap"].iloc[ruta[0]]
                if max(ruta[2]-data_2["Hora"].iloc[ruta[0]], 0) > 0 and data_2["Estimacion"].iloc[ruta[0]] == 0 and data_2["Turno"].iloc[ruta[0]] == 0:
                    data_2["Tiempo Stockout (horas)"].iloc[ruta[0]] += max(ruta[2]-data_2["Hora"].iloc[ruta[0]], 0)
                    sol2["Horas en Stockout"] += max(ruta[2]-data_2["Hora"].iloc[ruta[0]], 0)
                data_2["Tiempo Stockout"].iloc[ruta[0]] = 0
            data_2["Visitado"].iloc[ruta[0]] += 1

    data_1["Cantidad"] = np.maximum(cantidad_1, np.zeros(len(data_1)))
    data_2["Cantidad"] = np.maximum(cantidad_2, np.zeros(len(data_2)))
    data_1["Tiempo Stockout"].iloc[np.where(np.array(data_1["Cantidad"]) == 0)] += 1
    data_2["Tiempo Stockout"].iloc[np.where(np.array(data_2["Cantidad"]) == 0)] += 1
    data_1["Tiempo Stockout Total"].iloc[np.where(np.array(data_1["Cantidad"]) == 0)] += 1
    data_2["Tiempo Stockout Total"].iloc[np.where(np.array(data_2["Cantidad"]) == 0)] += 1
    data_1["Tiempo Stockout (horas)"].iloc[np.intersect1d(np.where(np.array(data_1["Cantidad"]) == 0), np.where(np.array(data_1["Tiempo Stockout"]) == 1))] += 8 - data_1["Hora"].iloc[np.intersect1d(np.where(np.array(data_1["Cantidad"]) == 0), np.where(np.array(data_1["Tiempo Stockout"]) == 1))]
    data_1["Tiempo Stockout (horas)"].iloc[np.intersect1d(np.where(np.array(data_1["Cantidad"]) == 0), np.where(np.array(data_1["Tiempo Stockout"]) != 1))] += 8
    sol1["Horas en Stockout"] += sum(8*np.ones(len(np.intersect1d(np.where(np.array(data_1["Cantidad"]) == 0), np.where(np.array(data_1["Tiempo Stockout"]) == 1)))) - np.array(data_1["Hora"].iloc[np.intersect1d(np.where(np.array(data_1["Cantidad"]) == 0), np.where(np.array(data_1["Tiempo Stockout"]) == 1))]))
    sol2["Horas en Stockout"] += sum(8*np.ones(len(np.intersect1d(np.where(np.array(data_2["Cantidad"]) == 0), np.where(np.array(data_2["Tiempo Stockout"]) == 1)))) - np.array(data_2["Hora"].iloc[np.intersect1d(np.where(np.array(data_2["Cantidad"]) == 0), np.where(np.array(data_2["Tiempo Stockout"]) == 1))]))
    sol1["Horas en Stockout"] += 8*len(np.intersect1d(np.where(np.array(data_1["Cantidad"]) == 0), np.where(np.array(data_1["Tiempo Stockout"]) != 1)))
    sol2["Horas en Stockout"] += 8*len(np.intersect1d(np.where(np.array(data_2["Cantidad"]) == 0), np.where(np.array(data_2["Tiempo Stockout"]) != 1)))
    data_1["Caida stockout"].iloc[np.intersect1d(np.where(np.array(data_1["Cantidad"]) == 0), np.where(np.array(data_1["Tiempo Stockout"]) == 1))] += 1
    data_2["Caida stockout"].iloc[np.intersect1d(np.where(np.array(data_2["Cantidad"]) == 0), np.where(np.array(data_2["Tiempo Stockout"]) == 1))] += 1
    sol1["Caidos en Stockot"] = len(np.intersect1d(np.where(np.array(data_1["Cantidad"]) == 0), np.where(np.array(data_1["Tiempo Stockout"]) == 1)))
    sol2["Caidos en Stockot"]  = len(np.intersect1d(np.where(np.array(data_2["Cantidad"]) == 0), np.where(np.array(data_2["Tiempo Stockout"]) == 1)))
    estimacion_stockout.estimator(data_1)
    estimacion_stockout.estimator(data_2)

    data_1.sort_values(by=combinacion, inplace=True, ascending=orden, na_position="first")
    data_2.sort_values(by=combinacion, inplace=True, ascending=orden, na_position="first")
    """
    indices1 = sort_random(data_1)
    ind1 = indices1.index(0)
    indices1.pop(ind1)
    data_1 = data_1.iloc[[0]+indices1]
    indices2 = sort_random(data_2)
    ind2 = indices2.index(0)
    indices2.pop(ind2)
    data_2 = data_2.iloc[[0]+indices2]
    """

    return data_1, data_2


def main():

    combinacion = ["Estimacion", "Turno"]
    orden = [True, True]
    data = read_data(combinacion, orden)
    data_1 = data.copy(deep=True)
    data_2 = data.copy(deep=True)

    simulaciones = 500
    soluciones = dict()
    soluciones["sol 1"] = list()
    soluciones["sol 2"] = list()

    for i in range(simulaciones):
        print(i)
        solution_proximity = proximity_approach.main(data_1)
        solution_stockout = stockout_approach.main(data_2) 
        solution_proximity["index"] = data_1.index.values.tolist()
        solution_stockout["index"] = data_2.index.values.tolist()
        data_1, data_2 = data_model(data_1, data_2, solution_proximity, solution_stockout, combinacion, orden)

        soluciones["sol 1"].append(solution_proximity)
        soluciones["sol 2"].append(solution_stockout)

    writer = pd.ExcelWriter('ranking.xlsx', engine='xlsxwriter')
    data_1.to_excel(writer, sheet_name='Solucion 1')
    data_2.to_excel(writer, sheet_name='Solucion 3')
    writer.save()

    with open(f'ranking.json', 'w') as file:
        json.dump(soluciones, file, indent=4)

if __name__ == '__main__':

    main()
 

