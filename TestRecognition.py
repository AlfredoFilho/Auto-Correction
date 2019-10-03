#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as pt

def digitsRecognition():
    data = pd.read_csv('data/train.csv')
    print(str(data.shape))

digitsRecognition()