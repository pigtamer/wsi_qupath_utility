"""[summary]
For divide the dataset into train_val_test
The volume is planned as 9:1:1
    10 fold validation with train and val
    1 for final test (is it necessary? we have Kimura samples already)
"""
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
