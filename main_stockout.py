#!/usr/bin/env python
# This Python file uses the following encoding: utf-8
# Copyright 2015 Tin Arm Engineering AB
# Copyright 2018 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Capacitated Vehicle Routing Problem with Time Windows (CVRPTW).
   This is a sample using the routing library python wrapper to solve a CVRPTW
   problem.
   A description of the problem can be found here:
   http://en.wikipedia.org/wiki/Vehicle_routing_problem.
   Distances are in meters and time in minutes.
"""

from __future__ import print_function

from functools import partial
from six.moves import xrange

from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

import numpy as np
import pandas as pd
from sklearn.neighbors import DistanceMetric
import estimacion_stockout


###########################
###  Lectura de Datos   ###
###########################


def time_windows_generator(num_locations):
    np.random.seed(2895)
    time_windows = list()

    for i in range(num_locations):
        a = np.random.randint(0, 479)
        b = np.random.randint(a, 480)
        time_windows.append([0, 480])
    return time_windows


###########################
# Problem Data Definition #
###########################
def create_data_model(atm_information):

    """Stores the data for the problem"""
    capacidad_maxima = 20
    data = {}
    data['locations'] = list(atm_information['Ubicacion'])
    data['num_locations'] = len(data['locations'])
    data['demands'] = list(atm_information['Gap'])
    data['num_vehicles'] = 5
    data['vehicle_capacity'] = 8*capacidad_maxima
    data['vehicle_speed'] = 830 # Velocidad de viaje: 50km/h
    data['depot'] = 0
    data['time_windows'] = time_windows_generator(data['num_locations'])
    data['estimacion'] = list(atm_information['Estimacion'])
    data['turno'] = list(atm_information['Turno'])
    return data


#######################
# Problem Constraints #
#######################


def create_distance_evaluator(data):
    """Creates callback to return distance between points."""

    # precompute distance between location to have distance callback in O(1)
    ubicaciones = np.radians(data['locations'])
    dist = DistanceMetric.get_metric('haversine')
    matriz = dist.pairwise(ubicaciones)*6373*1000
    _distances = np.int64(matriz)
    
    def distance_evaluator(manager, from_node, to_node):
        """Returns the manhattan distance between the two nodes"""
        return _distances[manager.IndexToNode(from_node), manager.IndexToNode(
            to_node)]

    return distance_evaluator


def create_demand_evaluator(data):
    """Creates callback to get demands at each location."""
    _demands = data['demands']

    def demand_evaluator(manager, node):
        """Returns the demand of the current node"""
        return _demands[manager.IndexToNode(node)]

    return demand_evaluator


def add_capacity_constraints(routing, data, demand_evaluator_index):
    """Adds capacity constraint"""
    capacity = 'Capacity'
    routing.AddDimension(
        demand_evaluator_index,
        0,  # null capacity slack
        data['vehicle_capacity'],
        True,  # start cumul to zero
        capacity)


def create_time_evaluator(data):
    """Creates callback to get total times between locations."""

    # precompute total time to have time callback in O(1)
    ubicaciones = np.radians(data['locations'])
    dist = DistanceMetric.get_metric('haversine')
    matriz = dist.pairwise(ubicaciones)*6373*1000/data['vehicle_speed']
    _total_time = np.int64(matriz)

    def time_evaluator(manager, from_node, to_node):
        """Returns the total time between the two nodes"""
        return _total_time[manager.IndexToNode(from_node), manager.IndexToNode(
            to_node)]

    return time_evaluator


def add_time_window_constraints(routing, manager, data, time_evaluator_index):
    """Add Global Span constraint"""
    time = 'Time'
    horizon = 480
    routing.AddDimension(
        time_evaluator_index,
        0,  # allow waiting time
        horizon,  # maximum time per vehicle
        False,  # don't force start cumul to zero since we are giving TW to start nodes
        time)
    time_dimension = routing.GetDimensionOrDie(time)
    # Add time window constraints for each location except depot
    # and 'copy' the slack var in the solution object (aka Assignment) to print it
    for location_idx, time_window in enumerate(data['time_windows']):
        if location_idx == 0:
            continue
        index = manager.NodeToIndex(location_idx)
        time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])
        routing.AddToAssignment(time_dimension.SlackVar(index))
    # Add time window constraints for each vehicle start node
    # and 'copy' the slack var in the solution object (aka Assignment) to print it
    for vehicle_id in xrange(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        time_dimension.CumulVar(index).SetRange(data['time_windows'][0][0],
                                                data['time_windows'][0][1])
        routing.AddToAssignment(time_dimension.SlackVar(index))
        # Warning: Slack var is not defined for vehicle's end node
        # routing.AddToAssignment(time_dimension.SlackVar(self.routing.End(vehicle_id)))


###########
# Printer #
###########
def print_solution(data, manager, routing, assignment):  # pylint:disable=too-many-locals
    """Prints assignment on dataframe"""
    solution = dict()

    capacity_dimension = routing.GetDimensionOrDie('Capacity')
    time_dimension = routing.GetDimensionOrDie('Time')

    dropped = 0
    for node in range(routing.Size()):
        if routing.IsStart(node) or routing.IsEnd(node):
            continue
        if assignment.Value(routing.NextVar(node)) == node:
                dropped += 1

    solution["Objective"] = 0
    solution["Visitados"] = data["num_locations"] - dropped

    for vehicle_id in xrange(data['num_vehicles']):
        solution[f"Vehicle {vehicle_id}"] = dict()
        solution[f"Vehicle {vehicle_id}"]["Route"] = list()
        solution[f"Vehicle {vehicle_id}"]["Total distance"] = 0
        solution[f"Vehicle {vehicle_id}"]["Total load"] = 0
        solution[f"Vehicle {vehicle_id}"]["Total time"] = 0

        index = routing.Start(vehicle_id)
        # plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        distance = 0
        while not routing.IsEnd(index):
            load_var = capacity_dimension.CumulVar(index)
            time_var = time_dimension.CumulVar(index)
            """slack_var = time_dimension.SlackVar(index)"""
            solution[f"Vehicle {vehicle_id}"]["Route"].append((manager.IndexToNode(index), assignment.Value(load_var), assignment.Min(time_var)/60))
            """plan_output += ' {0} Load({1}) Time({2},{3}) Slack({4},{5}) ->'.format(
                manager.IndexToNode(index),
                assignment.Value(load_var),
                assignment.Min(time_var),
                assignment.Max(time_var),
                assignment.Min(slack_var), assignment.Max(slack_var))"""
            previous_index = index
            index = assignment.Value(routing.NextVar(index))
            distance += routing.GetArcCostForVehicle(previous_index, index,
                                                     vehicle_id)
        load_var = capacity_dimension.CumulVar(index)
        time_var = time_dimension.CumulVar(index)
        """slack_var = time_dimension.SlackVar(index)"""
        solution[f"Vehicle {vehicle_id}"]["Route"].append((manager.IndexToNode(index), assignment.Value(load_var),  assignment.Min(time_var)/60))
        """plan_output += ' {0} Load({1}) Time({2},{3})\n'.format(
            manager.IndexToNode(index),
            assignment.Value(load_var),
            assignment.Min(time_var), assignment.Max(time_var))"""
        solution[f"Vehicle {vehicle_id}"]["Total distance"] = distance
        solution["Objective"] += distance
        solution[f"Vehicle {vehicle_id}"]["Total load"] = assignment.Value(load_var)
        solution[f"Vehicle {vehicle_id}"]["Total time"] = assignment.Value(time_var)

    return solution


########
# Main #
########
def main(dataframe):
    """Entry point of the program"""
    # Instantiate the data problem.
    i = 2

    while i < len(dataframe):

        atm_information = dataframe.iloc[:i]
        data = create_data_model(atm_information)

        # Create the routing index manager
        manager = pywrapcp.RoutingIndexManager(data['num_locations'],
                                            data['num_vehicles'], data['depot'])

        # Create Routing Model
        routing = pywrapcp.RoutingModel(manager)

        # Define weight of each edge
        distance_evaluator_index = routing.RegisterTransitCallback(
            partial(create_distance_evaluator(data), manager))
        routing.SetArcCostEvaluatorOfAllVehicles(distance_evaluator_index)

        # Add Capacity constraint
        demand_evaluator_index = routing.RegisterUnaryTransitCallback(
            partial(create_demand_evaluator(data), manager))
        add_capacity_constraints(routing, data, demand_evaluator_index)

        # Allow to drop nodes by capacity == TODOS TIENEN DEMANDA
        penalty = 1000000
        for node in range(1, data['num_locations']):
            routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

        # Add Time Window constraint
        time_evaluator_index = routing.RegisterTransitCallback(
            partial(create_time_evaluator(data), manager))
        add_time_window_constraints(routing, manager, data, time_evaluator_index)

        # Setting first solution heuristic (cheapest addition).
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.time_limit.seconds = 20
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)  # pylint: disable=no-member
        # Solve the problem.
        assignment = routing.SolveWithParameters(search_parameters)
        # assignment.time_limit.seconds = 30
        for node in range(routing.Size()):
            if routing.IsStart(node) or routing.IsEnd(node):
                continue
            if assignment.Value(routing.NextVar(node)) == node:
                solution = print_solution(data, manager, routing, assignment)
                stockout = 0
                while i < len(dataframe):
                    if (dataframe['Estimacion'].iloc[i-1] == 0 and dataframe['Turno'][i-1] == 0):
                        stockout += 1
                    i += 1
                solution["Stockout"] = stockout
                return solution
        i += 1

    solution = print_solution(data, manager, routing, assignment)
    return solution

