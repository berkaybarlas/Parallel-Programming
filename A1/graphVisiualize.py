import matplotlib.pyplot as plt
import os, sys
import time
import numpy as np
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator, FuncFormatter
import matplotlib

imgPath = './img/'

class FigureData:
    dataX = []
    dataY = []
    filename = ''
    title = ''
    color = 'g'
    style = '-o'
    xlabel = True

    def __init__(self, dataX, dataY, title, filename, color = 'g' ,style = '-o', xlabel = True):
        self.dataX = dataX
        self.dataY = dataY
        self.title = title
        self.filename = fileName
        self.color = color
        self.style = style
        self.xlabel = xlabel

    def plot(self):
        plt.clf()

        plt.ylabel('Speedup Factor')
        if self.xlabel:
            plt.xlabel('Number of Threads')
            tick_values = [2**x for x in range(0,6)]
            plt.xticks(tick_values,[("%.0f" % x)  for x in tick_values])
        plt.yticks(self.dataY,[("%.2f" % x)  for x in self.dataY])
        plt.title(self.title)
        plt.plot(self.dataX, self.dataY, self.style, color=self.color)
    
    def print(self):
        self.plot()
        plt.savefig(imgPath + self.filename + '.png')

    def show(self):
        self.plot()
        plt.show()
    

if not os.path.isdir(imgPath):
            os.mkdir(imgPath)
printMode = False

if len(sys.argv) > 1:
    if sys.argv[1] == '1':
        printMode = True


figureDatas = []
numCores = [1, 2, 4, 8, 16, 32]

# Data 1 #
serial_part1A = 13.33
speedup_part1A = [31.13, 19.60, 12.16, 9.08, 7.59, 6.79]
speedup_part1A = [serial_part1A / x for x in speedup_part1A]
fileName = 'speedup_part_1_A'
title ='Speedup Image Blurring Coffee'

partA_coffee = FigureData(numCores, speedup_part1A, title, fileName)
figureDatas.append(partA_coffee)
##

# Data 2 #
serial_part1A_strawberry = 25.24
speedup_part1A_strawberry = [55.99, 34.58, 22.54, 16.96, 13.72, 12.84]
speedup_part1A_strawberry = [serial_part1A_strawberry / x for x in speedup_part1A_strawberry]
title ='Speedup Image Blurring Strawberry'
fileName = 'speedup_part_1_A_strawberry'

partA_strawberry = FigureData(numCores, speedup_part1A_strawberry, title, fileName)
figureDatas.append(partA_strawberry)
##

# Part 1 B#
serial_part1B_coffee = 13.33
speedup_part1B_coffee = [7.59, 10.29]
speedup_part1B_coffee = [serial_part1B_coffee / x for x in speedup_part1B_coffee]
title =' Thread Binding Image Blurring Coffee'
fileName = 'binding_part_1_B_coffee'

partB_coffee = FigureData(['Compact','Scatter'], speedup_part1B_coffee, title, fileName, 'red', 'o', False)
figureDatas.append(partB_coffee)
##

# Part 1 B#
serial_part1B_strawberry = 25.24
speedup_part1B_strawberry = [13.72, 16.30]
speedup_part1B_strawberry = [serial_part1B_strawberry / x for x in speedup_part1B_strawberry]
title =' Thread Binding Image Blurring Strawberry'
fileName = 'binding_part_1_B_strawberry'

partB_strawberry = FigureData(['Compact','Scatter'], speedup_part1B_strawberry, title, fileName, 'red', 'o', False)
figureDatas.append(partB_strawberry)
##

# Data Sudoku hard3 A #
serial_part2A = 48.10
speedup_part2A = [78.69, 47.05, 26.63, 14.72, 7.42, 4.29]
speedup_part2A = [serial_part2A / x for x in speedup_part2A]
fileName = 'speedup_part_2_A'
title ='Speedup Sudoku Part A 4x4_hard3.csv'

partA_sudoku_speedup_hard3 = FigureData(numCores, speedup_part2A, title, fileName)
figureDatas.append(partA_sudoku_speedup_hard3)
##

# Data Sudoku hard3 B #
serial_part2A = 48.10
speedup_part2B = [74.50, 39.36, 21.32, 12.87, 6.58, 3.85]
speedup_part2B = [serial_part2A / x for x in speedup_part2B]
fileName = 'speedup_part_2_B'
title ='Speedup Sudoku Part B Cutoff=30 4x4_hard3.csv'

partB_sudoku_speedup_hard3 = FigureData(numCores, speedup_part2B, title, fileName)
figureDatas.append(partB_sudoku_speedup_hard3)
##

# Data Sudoku hard3 C #
serial_part2C = 0.33
speedup_part2C = [0.57, 0.63, 0.67, 0.75, 0.73, 0.31]
speedup_part2C = [serial_part2C / x for x in speedup_part2C]
fileName = 'speedup_part_2_C'
title ='Speedup Sudoku Part C 4x4_hard3,csv'

partC_sudoku_speedup_hard3 = FigureData(numCores, speedup_part2C, title, fileName)
figureDatas.append(partC_sudoku_speedup_hard3)
##

# Part 2 b) A#
bindging_part2A = [7.42, 7.52]
bindging_part2A = [serial_part2A / x for x in bindging_part2A]
title =' Thread Binding Sudoku Part A 4x4_hard3.csv'
fileName = 'binding_part_2_A'

partA_sudoku_binding = FigureData(['Compact','Scatter'], bindging_part2A, title, fileName, 'red', 'o', False)
figureDatas.append(partA_sudoku_binding)
##

# Part 2 b) B#
bindging_part2B = [6.58, 7.16]
bindging_part2B = [serial_part2A / x for x in bindging_part2B]
title =' Thread Binding Sudoku Part B 4x4_hard3.csv'
fileName = 'binding_part_2_B'

partB_sudoku_binding = FigureData(['Compact','Scatter'], bindging_part2B, title, fileName, 'red', 'o', False)
figureDatas.append(partB_sudoku_binding)
##

# Part 2 b) C#
bindging_part2C = [0.73, 0.75]
bindging_part2C = [serial_part2C / x for x in bindging_part2C]
title =' Thread Binding Sudoku Part C 4x4_hard3.csv'
fileName = 'binding_part_2_C'

partC_sudoku_binding = FigureData(['Compact','Scatter'], bindging_part2C, title, fileName, 'red', 'o', False)
figureDatas.append(partC_sudoku_binding)
##

# Part 2 c) B#
serial_hard1 = 23.68
serial_hard2 = 44.04
serial_hard3 = 47.39
speedup_32_core = [serial_hard1 / 1.72, serial_hard2 / 3.11 , serial_hard3 / 3.27]
#speedup_32_core = [serial_part2A / x for x in speedup_32_core]
title ='Sudoku with Different Grids Part B - with 32 Thread and Cutoff: 40'
fileName = 'grids_part_2_B'

partB_sudoku_32 = FigureData(['4x4_hard1','4x4_hard2','4x4_hard3'], speedup_32_core, title, fileName, 'red', 'o', False)
figureDatas.append(partB_sudoku_32)
##



for data in figureDatas:
    if printMode:        
        data.print()
    else:
        data.show()


