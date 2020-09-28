

class Sudoku_Solver:
    @staticmethod
    def check_valid(puzzle):
        numbers = 0
        for row_index in range(9):
            for col_index in range(9):
                num = puzzle[row_index][col_index]
                if num != 0:
                    numbers += 1
                    puzzle[row_index][col_index] = 0
                    res = Sudoku_Solver.check_safe(puzzle,row_index,col_index,num)
                    if not res:
                        return False
                    puzzle[row_index][col_index] = num
        
        if numbers > 16:
            return True
        else:
            return False

    @staticmethod
    def check_row(puzzle,row_index,num):
        for col_index in range(9):
            if puzzle[row_index][col_index] == num:
                return False

        return True

    @staticmethod
    def check_col(puzzle,col_index,num):
        for row_index in range(9):
            if puzzle[row_index][col_index] == num:
                return False

        return True

    @staticmethod
    def check_square(puzzle,row_index,col_index,num):
        row_index -= (row_index % 3)
        col_index -= (col_index % 3)

        for i in range(3):
            for j in range(3):
                if puzzle[row_index+i][col_index+j] == num:
                    return False

        return True

    @staticmethod
    def check_safe(puzzle,row_index,col_index,num):
        return Sudoku_Solver.check_row(puzzle,row_index,num) and Sudoku_Solver.check_col(puzzle,col_index,num)  and Sudoku_Solver.check_square(puzzle,row_index,col_index,num)

    @staticmethod
    def find_empty(puzzle):
        for row_index in range(9):
            for col_index in range(9):
                if puzzle[row_index][col_index] == 0:
                    return (row_index,col_index)

        return (-1,-1)

    @staticmethod
    def solve(puzzle):
        row_index, col_index = Sudoku_Solver.find_empty(puzzle)
        if row_index == -1 and col_index == -1:
            return True

        for num in range(1,10):
            if Sudoku_Solver.check_safe(puzzle,row_index,col_index,num):
                puzzle[row_index][col_index] = num

                if Sudoku_Solver.solve(puzzle):
                    return True

                puzzle[row_index][col_index] = 0

        return False
            
if __name__ == "__main__":
    puz = [[8, 0, 0, 0, 1, 0, 0, 0, 9], 
            [0, 5, 0, 8, 0, 7, 0, 1, 0], 
            [0, 0, 4, 0, 9, 0, 7, 0, 0], 
            [0, 6, 0, 7, 0, 1, 0, 2, 0], 
            [5, 0, 8, 0, 6, 0, 1, 0, 7], 
            [0, 1, 0, 5, 0, 2, 0, 9, 0], 
            [0, 0, 7, 0, 4, 0, 6, 0, 0], 
            [0, 8, 0, 3, 0, 9, 0, 4, 0], 
            [3, 0, 0, 0, 5, 0, 0, 0, 8]]

    print(Sudoku_Solver.check_valid(puz))
    # res = Sudoku_Solver.solve(puz)
    # for row in puz:
    #     print(row)