Sudoku Solver OpenCV & Tensorflow
=================================

Introduction
------------

Solves sudoku puzzle given as input and overlay the result to the image.

## Requirements

Check requirements.txt file

## Usage

- By default the program uses a pre-trained model for digit recognition. If you want to train your own model you
 can do so by running.

 > python digit_classifier.py --train model_name

- To solve sudoku puzzle from image run

 > python main.py -i image_path

- To specify the model to be used use the -m argument when running main.py

 > python main.py -i image_path -m model_path
