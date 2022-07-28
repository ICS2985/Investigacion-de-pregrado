import pandas as pd
import math as mt
import numpy as np
# Estamos utilizando la demanda de los cajeros como el
# gap entre la demanda y la oferta.
# Estoy pensando en las dimensiones de la
# capacidad del cajero y la demanda ~N

# Esto funciona asumiendo misma demanda en todos los tiempos
def estimator(dataframe):
    tiempos = list()
    turnos = list()
    gaps = np.array(dataframe["Demanda"]) - np.array(dataframe["Cantidad"])
    gaps[0] = 0
    horas = ((np.array(dataframe["Cantidad"])/(np.array(dataframe["Demanda"])/24))%24)%8
    for gap in range(1, len(gaps)):
        d = np.array(dataframe["Demanda"])[gap]/3
        if gaps[gap] > 0:   # quedo en stockout hoy
            tiempos.append(0)
            gap_turn = d - np.array(dataframe["Cantidad"])[gap]
            if gap_turn > 0: # quedo en stockout turno 0 de hoy
                turnos.append(0)
            else:  # quedo en stockout turno 1 y 2 de hoy
                turno = mt.ceil(abs(gap_turn/d))
                turnos.append(turno)
            gaps[gap] = 20
        else: # quedo en stockout el resto de los días
            t = abs(gaps[gap]/np.array(dataframe["Demanda"])[gap]) # calculo el día que quedo stockout
            tiempo = mt.ceil(t)
            tiempos.append(tiempo)

            gap_turn = d - abs(gaps[gap]) % np.array(dataframe["Demanda"])[gap] # calculo el turno del día que quedo stock out
            if gap_turn > 0: # stockout primer turno
                turnos.append(0)
            else: # stockout turno 1 y 2
                turno = mt.ceil(abs(gap_turn/d))
                turnos.append(turno)

            if tiempo < 2:
                gaps[gap] = 20
            else:
                if abs(gaps[gap]) < 8: # si lo que me queda en el cajero es menor al casette
                    gaps[gap] = 8
                else:
                    gaps[gap] = 20
    dataframe["Estimacion"] = [0] + tiempos
    dataframe["Turno"] = [0] + turnos
    dataframe["Gap"] = gaps
    dataframe["Hora"] = horas
    return


def stockout_by_period(Estimacion):
    stockout = dict()
    max_period = max(Estimacion)
    for i in range(0, max_period + 1):
        count = Estimacion.count(i)
        if count > 0:
            stockout[i] = count
    return stockout
