#!/usr/bin/env python
# coding: utf-8

force_production = False #True

import math
import os
import pickle

import networkx as nx
import numpy as np

# Default configuration
from configuration import *

# # Introduction
# 
# The global launcher will launch a series of algorithms and save their result in pickle files.
# It supports:
# - Threshold
# - Multiple matrices (weight, binary)
# - Multiple additional parameters
# 
# Algorithm took as input a tuple:
# (module, algorithm, matrix_mask, param1, ..., paramN)
# 
# matrix_mask is a string composed by:
# - 'o' for original
# - 'b' for binary
# - 'w' for weight
# - 'g' for networkx Graph, computed on 'w'
# 
# Example (with threshold 0.1):
# ('bct', 'betweeness', ['b']) will produce execution of
# `bct.betweeness(binary_adj)` and the creation of file bct-betweeness-b-0.1.pickle
# 
# ('bct', 'latmio_und', ['w'], 1) will produce execution of
# `bct.latmio_und(weighted_adj, 1)` and the creation of file bct-latmio_und-w-0.1.pickle

# # Matrix Utility Class
class Matrix:
    def __init__(self, matrix):
        self.matrix = np.copy(matrix)
    
    """
    Make a graph starting from a weighted matrix
    """
    # @lru_cache(maxsize=1)
    def _make_graph(self):
        return nx.convert_matrix.from_numpy_matrix(self.matrix)

    def get_matrix(self):
        return self.matrix
    
    """
    Binarize a matrix based on a threshold.
    """
    # @lru_cache(maxsize=10)
    def get_binary_matrix(self, threshold):
        return np.where(np.abs(self.matrix) < threshold, 0, 1)

    # @lru_cache(maxsize=10)
    def get_weighted_matrix(self, threshold):
        adj_matrix = np.copy(self.matrix)
        adj_matrix[np.absolute(adj_matrix) < threshold] = 0
        return adj_matrix
    
    """
    type is bin for binary, wei for weighted
    """
    def get_matrix_by_type(self, type_name, threshold):
        if type_name == "b":
            return self.get_binary_matrix(threshold)
        elif type_name == "w":
            return self.get_weighted_matrix(threshold)   
        elif type_name == "o":
            return self.get_matrix()   
        elif type_name == "g":
            return nx.Graph(self.get_weighted_matrix(threshold))
        return None

# # Naming
# 
# Each file report the following informations:
# - subject_id: Subject identifier
# - module: module that contains algorithm
# - algorithm: function to compute the metric
# - matrix: correlation matrix
# - threshold: Threshold to create weighted and binary matrices
# - matrix mask: Mask to create a list of binary or weighted matrices
# - *params: additional parameters

def compute(subject_dir, subject_id, module_name, algorithm, matrix: Matrix, threshold, matrix_mask, *params):

    module = __import__(module_name)
    function = getattr(module, algorithm)

    inputs = []

    for type in matrix_mask:
        inputs.append(matrix.get_matrix_by_type(type, threshold))

    for param in params:
        inputs.append(param)
   	
    try: 
        return function(*inputs)
    except:
        return None
    return None


"""
get_file_name produces a path where save result of an algorithm's exacution
"""
def get_file_name(subject_dir, subject_id, index, threshold, matrix_mask, module_name, algorithm, *params):
    # Compute file name
    file_name = f"{subject_id}-{index}-{threshold}-{matrix_mask}-{module_name}-{algorithm}"
    for param in params:
        file_name = file_name + "-" + str(param)
    return f"{subject_dir}/{subject_id}/{DIR_RESULTS}/{file_name}.pickle"

def to_string(functions):
    result = ""
    for function in functions:
        ( module_name, algorithm, matrix_mask, *params) = function
        result = result + algorithm
    return result

def compute_and_save(subject_dir, subject_ids, indexes, thresholds, functions):

    try:
        index = "-"
        for subject_id in subject_ids:

            os.makedirs(f"{subject_dir}/{subject_id}/{DIR_RESULTS}", exist_ok=True)

            for index in indexes:

                # Load correlation matrix
                with open(f"{subject_dir}/{subject_id}/{DIR_CORRELATION}/{index}.pickle", "rb") as f:
                    matrix = pickle.load(f)

                m = Matrix(matrix)

                for threshold in thresholds:

                    # Compute each function and save it properly 
                    for function in functions:

                        # Load data
                        (module_name, algorithm, matrix_mask, *params) = function

                        # Compute file name
                        file_name = get_file_name(subject_dir, subject_id, index, threshold, matrix_mask, module_name, algorithm, *params)
                        print(f"Producing: {file_name}",flush=True)
                        if force_production or not os.path.exists(file_name):
                            # Obtain result and save it
                            result = compute(subject_dir, subject_id, module_name, algorithm, m, threshold, matrix_mask, *params)
                            if result is None:
                                print(f"Skip: {file_name}")
                            with open(file_name, 'wb') as handle:
                                pickle.dump(result, handle)

    except:
        print("Blocco Mortale:", index, threshold, to_string(functions))

"""
Dump informations about a single file
"""
def dump_info(path):
    file_name = os.path.basename(path.replace(".pickle", ""))
    (subject_id, index, threshold, matrix_mask, module, algorithm, *params) = file_name.split("-")
    m = {
        'b': 'Binary',
        'w': 'Weighted'
    }
    print("%12s: %s" % ("Subject ID", subject_id))
    print("%12s: %s" % ("Index", index))
    print("%12s: %s" % ("Threshold", threshold))
    print("%12s: %s" % ("Matrices", [m[x] for x in "bw"]))
    print("%12s: %s" % ("Module", module))
    print("%12s: %s" % ("Algorithm", algorithm))
    if len(params) > 0:
        print("%12s: %s" % ("Params", params))

# Simple test
# dump_info("0040013-0-0.1-b-bct-betweenness_bin-0.1-0.2.pickle")
# dump_info("0040013-0-[0.1]-w-luciani-bonachic_centrality_und-0.5.pickle")

print("#-------------------")
print("# Centrality")
print("#-------------------")

phi = (1 + math.sqrt(5))

compute_and_save(subjects_dir,
                [ subject_id ],
                indexes,
                thresholds,
                [
                    
                    ('luciani', 'betweenness_wei', 'w'),
                    
                    
                    ('bct', 'eigenvector_centrality_und', 'w'),
                   
                ])

# %%time

print("#-------------------")
print("# Clustering")
print("#-------------------")

compute_and_save(subjects_dir,
                [ subject_id ],
                indexes,
                thresholds,
                [
                   
                    ('bct', 'clustering_coef_wu', 'w'),
                    ('bct', 'transitivity_wu', 'w'),
                    #bct.transitivity_bd(bin_matrix)  escono valori pari a inf
                   
                ])

# %%time



# %%time

print("#-------------------")
print("# Degree")
print("#-------------------")

compute_and_save(subjects_dir,
                [ subject_id ],
                indexes,
                thresholds,
                [
                    ('bct', 'degrees_und', 'w'),

                ])


