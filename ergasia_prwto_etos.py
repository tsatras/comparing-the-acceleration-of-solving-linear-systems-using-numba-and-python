import tkinter as tk


import numpy as np
from numba import njit
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pylab as plt
import scipy.sparse as sparse
import time

#The following function is a jitted function using numba. By using the njit, the function that followes can not access all the python libraries
#as numba can do only simple operations
@njit
def solve_linear_system_numba(A, b):
    x = np.linalg.solve(A, b)
    return x

np.random.seed(42)
A = np.random.rand(1000, 1000)
b = np.random.rand(1000)

#run the function one time in order to compile the whole process, so that numba can be more effective when we would like to use it
compilation = solve_linear_system_numba(A , b)

#the following function creates a matrix of 1000x1000 dimensions and density of 0.01. After that, it generates a vector with 1000 indices that is used to solve the linear system
cols = 1000
rows = 1000
density = 0.01

A_matrix = csc_matrix(np.random.rand(cols , rows) < density)

B_vector = np.random.rand(cols)


def solve_using_python():
    y = spsolve(A_matrix , B_vector)
    return y


# Sparse Matrix Visualization

# create a sparse matrix with specific density
def matrix_visualization():
    A = sparse.random(1000,1000, density=0.0001)
    plt.spy(A, markersize=4)
    # visualize the sparse matrix with Spy
    plt.spy(A)
    plt.show()




class Sparse_interface():
    def __init__(self,root):
        self.w = root
        self.w.geometry('694x400')
        self.w.title('Project Python')
        self.label = tk.Label(text = 'Sparse Matrices Solvers',font = 'Bahnschrift 32')
        self.label.pack()
        self.button1 = tk.Button(text = 'Solve with Python',font ='Bahnschrift 14',bg='white', command=self.solve_python)
        self.button1.place(relx=0.5,rely=0.55,anchor=tk.CENTER)
        self.button2 = tk.Button(text = 'Solve with Numba', font='Bahnschrift 14',bg='white', command = self.solve_numba)
        self.button2.place(relx=0.5,rely=0.68,anchor=tk.CENTER)
        self.button3 = tk.Button(text = 'Compare Solvers',font = 'Bahnschrift 15',bg='white', command = self.compare_solvers)
        self.button3.place(relx=0.5,rely=0.81,anchor=tk.CENTER)
        self.button4 = tk.Button(text='Exit',font='Bahnschrift 13',bg='white',command=self.exit)
        self.button4.place(relx=0,rely=0)
        self.button5 = tk.Button(text='Matrices Visualization',font = 'Bahnschrift 13',bg='white',command = self.visualize)
        self.button5.place(relx=0.5,rely=0.94,anchor = tk.CENTER)


    def solve_python(self):
        start_time = time.time()
        solution = solve_using_python()
        end_time = time.time()

        print(solution , '\n', 'estimated time to solve linear system using python method is: {}'.format(end_time - start_time), 'seconds' )


    def solve_numba(self):
        start = time.time()
        solution_x = solve_linear_system_numba(A,b)
        end = time.time()
        print(solution_x,'\n', 'estimated time to solve linear system using numba method is:{}'.format(end - start), 'seconds')


    def exit(self):
        self.w.destroy()
        
    def visualize(self):
        matrix_visualization()

    def compare_solvers(self):
        start_time = time.time()
        solve_using_python()
        end_time = time.time()

        start = time.time()
        solve_linear_system_numba(A,b)
        end = time.time()

        print('solving the linear system using numba was: {}'.format((end_time - start_time) - (end - start)),'seconds faster, than solving it using python')
        


root = tk.Tk()
Sparse_interface(root)
root.mainloop()
