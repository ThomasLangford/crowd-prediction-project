"""Entry point of the code demonstration for the project.

This file is the entry point for running the short demonstration acompanying
the project report. The parametaers in this file should not be changed unless
they accompany an update to the model, as changes to the frame interval and the
buffer length will prevent the prediction model from functioning normally.
"""
from visualisation.online import online

image_directory = "./example"

frame_interval = 1
buffer_length = 8
prediction_length = 10
error_threshold = 5

online(image_directory, frame_interval, buffer_length, prediction_length,
       error_threshold)
