import numpy as np
import os
import matplotlib.pyplot as plt
import json
import pandas as pd
import re
import copy

def augmentation_y_shift(array_x, 
                         array_y, 
                         shift_value = 5, 
                         x_lim = 80):
    """_summary_

    Args:
        array_x (_type_): _description_
        array_y (_type_): _description_
        shift_value (_type_, optional): _description_. Defaults to np.random.randint(-5, 5).
        percentage_added (float, optional): How much of array_x gets shifted. Defaults to 0.5.
        percentage_skipped (float, optional): How much of array_x does not get augmented. Defaults to 0.5.
    """
    
    #focusing on skipped only.
    new_array_x = []
    for trial in array_x:
        trial["x"] += np.random.randint(-shift_value, shift_value)
        trial = np.delete(trial, 
                            list(np.where(trial["x"] >= x_lim)[0]) + \
                            list(np.where(trial["x"] < 0)[0]))
        new_array_x.append(trial) 
    return new_array_x, array_y
        
def merge_and_return_a_new(array_x,
                           array_y,
                           x_lim = 1600,
                           y_lim = 80,
                           percentage_added = 0.5):
    """Augmentation for combining two images and returning a 'merged' combined dataset

    Args:
        array_x (_type_): spike array (formatted as a [[(x, t, p)...spike times...]...array....])
        array_y (_type_): y label for array x
        x_lim (int, optional): x limit for neuron range within layer. Defaults to 1600.
        y_lim (int, optional): y limit for spike time range within trial. Defaults to 80.
        percentage_added (float, optional): how many new merged images are appended to dataset, 50% = 50% increase when returned. Defaults to 0.5.
    """
    
    new_array_x = copy.deepcopy(array_x)
    new_array_y = copy.deepcopy(array_y)
    
    appended_array_x = []
    appended_array_y = []
    
    # loop through all categories
    for category in np.unique(new_array_y):
        
        # percentage added (0.5 adds 50%, 1.0 adds 100% ...)
        for i in range(int(np.where(new_array_y == category)[0].shape[0] * percentage_added)):
            
            # checking unique images to merge
            image_1_id, image_2_id = 0, 0
            while image_1_id == image_2_id:
                image_1_id = np.where(new_array_y == category)[0][np.random.randint(0, np.where(new_array_y == category)[0].shape[0])]
                image_2_id = np.where(new_array_y == category)[0][np.random.randint(0, np.where(new_array_y == category)[0].shape[0])]
                
            # shifting along x axis by half the difference for each image    
            x1 = new_array_x[image_1_id]
            x2 = new_array_x[image_2_id]
            x1["t"] -= int((np.sum(x1["t"]) / 
                                                 x1["t"].shape[0] - np.sum(x2["t"]) / 
                                                 x2["t"].shape[0]) / 2)
            
            x2["t"] += int((np.sum(x1["t"]) / 
                                                 x1["t"].shape[0] - np.sum(x2["t"]) / 
                                                 x2["t"].shape[0]) / 2)

            # deleting out of bounds shifted times
            x1 = np.delete(x1, 
                                list(np.where(x1["t"] > x_lim)[0]) + \
                                list(np.where(x1["t"] < 0)[0]))
            x2 = np.delete(x2, 
                                list(np.where(x2["t"] > x_lim)[0]) + \
                                list(np.where(x2["t"] < 0)[0]))
            
            # merge both images by skipping every other spike time, followed by sorting by spike time
            x = np.sort(np.concatenate((x1[np.random.uniform(size=len(x1)) < 0.5], x2[np.random.uniform(size=len(x2)) < 0.5])), order = "t")
            
            appended_array_x.append(x)
            appended_array_y.append(category)  
    
    return (new_array_x + appended_array_x, 
            new_array_y + appended_array_y)


def neuron_dropout(array_x, array_y, num_neurons, p_drop):
    new_x = copy.deepcopy(array_x)
    for i in range(len(new_x)):
       drop_neuron = list(np.where(np.random.uniform(size=num_neurons) < p_drop)[0])
       new_x[i]  = np.delete(new_x[i], [ i for i,x in enumerate(new_x[i]["x"]) if x in drop_neuron])

    return new_x, array_y
