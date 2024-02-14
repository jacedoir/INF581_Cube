from itertools import combinations
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection



class Cube:
    """A class to represent a Rubik's cube.
        color convention:
        - 0: white
        - 1: yellow
        - 2: green
        - 3: blue
        - 4: red
        - 5: orange
    """

    def __init__(self, size):
        self.size = size
        self.color_dict = {0: 'white', 1: 'yellow', 2: 'green', 3: 'blue', 4: 'red', 5: 'orange'}
        self.face_name_dict = {0: 'top', 1: 'left', 2: 'front', 3: 'right', 4: 'back', 5: 'bottom'}
        top = np.zeros((size, size))
        bottom = np.array([[1]*size]*size)
        front = np.array([[4]*size]*size)
        right = np.array([[3]*size]*size)
        back = np.array([[5]*size]*size)
        left = np.array([[2]*size]*size)
        self.state = np.array([top, left, front, right, back, bottom])
    
    def horizontal_twist(self, row, direction):
        """Twist a horizontal row of the cube.
        Args:
            row: int, the row to twist
            direction: int, 1 for clockwise, -1 for counterclockwise
        """
        try:
            assert row < self.size
            if direction == 1: # clockwise
                self.sate[1][row], self.sate[2][row], self.state[3][row], self.state[4][row] = self.state[2][row],self.state[3][row],self.state[4][row],self.state[1][row]
            else: # counterclockwise
                self.cube[1][row], self.cube[2][row], self.cube[3][row], self.cube[4][row] = self.cube[4][row],self.cube[1][row], self.cube[2][row], self.cube[3][row]
            
            if direction == 1: #clockwise
                if row == 0:
                    self.state[0] = [list(x) for x in zip(*reversed(self.state[0]))] #Transpose top
                elif row == len(self.state[0]) - 1:
                    self.state[5] = [list(x) for x in zip(*reversed(self.state[5]))] #Transpose bottom
            else: #counterclockwise
                if row == 0:
                    self.state[0] = [list(x) for x in zip(*self.state[0])][::-1] #Transpose top
                elif row == len(self.state[0]) - 1:
                    self.state[5] = [list(x) for x in zip(*self.state[5])][::-1] #Transpose bottom
        except:
            print("Invalid row number")

    def vertical_twist(self, column, direction):
        """Twist a vertical column of the cube.
        Args:
            column: int, the column to twist
            direction: int, 1 for clockwise, -1 for counterclockwise
        """
        try:
            assert column < self.size
            if direction == 1: # clockwise
                self.state[0][:,column], self.state[2][:,column], self.state[5][:,column], self.state[4][:,column] = self.state[4][:,column],self.state[0][:,column],self.state[2][:,column],self.state[5][:,column]
            else: # counterclockwise
                self.state[0][:,column], self.state[2][:,column], self.state[5][:,column], self.state[4][:,column] = self.state[2][:,column],self.state[5][:,column],self.state[4][:,column],self.state[0][:,column]
            
            if direction == 1: #clockwise
                if column == 0:
                    self.state[1] = [list(x) for x in zip(*reversed(self.state[1]))][::-1] #Transpose left
                elif column == len(self.state[1]) - 1:
                    self.state[3] = [list(x) for x in zip(*reversed(self.state[3]))] #Transpose right
            else: #counterclockwise
                if column == 0:
                    self.state[1] = [list(x) for x in zip(*self.state[1])][::-1] #Transpose left
                elif column == len(self.state[1]) - 1:
                    self.state[3] = [list(x) for x in zip(*self.state[3])] #Transpose right
        except:
            print("Invalid column number")

    def rotate_face(self, face, direction):
        """Rotate a face of the cube.
        Args:
            face: int, the face to rotate
            direction: int, 1 for clockwise, -1 for counterclockwise
        """
        if direction == 1:
            self.state[face] = [list(x) for x in zip(*reversed(self.state[face]))][::-1]
        else:
            self.state[face] = [list(x) for x in zip(*self.state[face])][::-1]


    def show_cube(self):
        """Show the current state of the cube, with color, for visualization."""
        #Build the 3D cube with the right colors
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Define the vertices of the cube
        r = [-1, 1]
        for s, e in combinations(np.array(list(combinations(r, 3))), 2):
            if np.sum(np.abs(s-e)) == r[1]-r[0]:
                ax.plot3D(*zip(s, e), color="b")

        # Define the colors of the faces
        colors = [self.color_dict[i] for i in range(6)]
        for i in range(6):
            ax.add_collection3d(Poly3DCollection([list(zip(*self.state[i]))], color=colors[i], linewidths=1, edgecolors='black', alpha=0.5))

        # Set the viewing angle
        ax.view_init(30, 30)
        plt.show()





       
cube_variable = Cube(3)
cube_variable.show_cube()