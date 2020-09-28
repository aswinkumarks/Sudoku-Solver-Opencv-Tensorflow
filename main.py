import argparse
from solver import Sudoku_Solver
from detector import Sudoku_Detector
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("-i","--image", required=True, help="path to input sudoku puzzle image")
parser.add_argument('-d',"--debug",default=False, type=bool)
args = parser.parse_args()

image = cv2.imread(args.image)
detector = Sudoku_Detector(debug=args.debug)
puzzle = detector.detect(image)
valid = Sudoku_Solver.check_valid(puzzle)
if not valid:
    print("Invalid puzzle")
    exit(0)

Sudoku_Solver.solve(puzzle)
detector.visualize_solution(puzzle)
