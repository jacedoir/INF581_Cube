from itertools import combinations
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection,Line3DCollection



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
        new_state = np.copy(self.state)
        try:
            assert row < self.size
            if direction == 1: # clockwise
                new_state[1][row], new_state[2][row], new_state[3][row], new_state[4][row] = self.state[2][row],self.state[3][row],self.state[4][row],self.state[1][row]
            else: # counterclockwise
                new_state[1][row], new_state[2][row], new_state[3][row], new_state[4][row] = self.state[4][row],self.state[1][row],self.state[2][row],self.state[3][row]
            
            if direction == 1: #clockwise
                if row == 0:
                    new_state[0] = [list(x) for x in zip(*reversed(self.state[0]))][::-1] #Transpose top
                elif row == len(self.state[0]) - 1:
                    new_state[5] = [list(x) for x in zip(*reversed(self.state[5]))][::-1] #Transpose bottom
            else: #counterclockwise
                if row == 0:
                    new_state[0] = [list(x) for x in zip(*self.state[0])][::-1]
                elif row == len(self.state[0]) - 1:
                    new_state[5] = [list(x) for x in zip(*self.state[5])][::-1]
            self.state = new_state.copy()
        except:
            print("Invalid row number")

    def vertical_twist(self, column, direction):
        """Twist a vertical column of the cube.
        Args:
            column: int, the column to twist
            direction: int, 1 for clockwise, -1 for counterclockwise
        """
        new_state = np.copy(self.state)
        try:
            assert column < self.size
            if direction == 1: # clockwise
                new_state[0][:, column], new_state[2][:, column], new_state[5][:, column], new_state[4][:, column] = self.state[4][:, column],self.state[0][:, column],self.state[2][:, column],self.state[5][:, column]
            else: # counterclockwise
                new_state[0][:, column], new_state[2][:, column], new_state[5][:, column], new_state[4][:, column] = self.state[2][:, column],self.state[5][:, column],self.state[4][:, column],self.state[0][:, column]
            if direction == 1: #clockwise
                if column == 0:
                    new_state[1] = [list(x) for x in zip(*reversed(self.state[1]))][::-1] #Transpose left
                elif column == len(self.state[1]) - 1:
                    new_state[3] = [list(x) for x in zip(*reversed(self.state[3]))][::-1] #Transpose right
            else: #counterclockwise
                if column == 0:
                    new_state[1] = [list(x) for x in zip(*self.state[1])][::-1]
                elif column == len(self.state[1]) - 1:
                    new_state[3] = [list(x) for x in zip(*self.state[3])][::-1]
            self.state = new_state.copy()
        except:
            print("Invalid column number")

    def rotate_face(self, face, direction):
        """Rotate a face of the cube.
        Args:
            face: int, the face to rotate
            direction: int, 1 for clockwise, -1 for counterclockwise
        """
        new_state = np.copy(self.state)
        if direction == 1:
            new_state[face] = [list(x) for x in zip(*reversed(self.state[face]))][::-1]
        else:
            new_state[face] = [list(x) for x in zip(*self.state[face])][::-1]
        self.state = new_state.copy()


    def show_cube_2D(self):
        """Show the current state of the cube, with color, for visualization."""
        fig, axs = plt.subplots(2, 3, figsize=(12, 8))
        for i in range(6):
            ax = axs[i//3, i%3]
            ax.imshow(self.state[i])
            ax.set_title(self.face_name_dict[i])
            ax.axis('off')
            for j in range(self.size):
                for k in range(self.size):
                    label = str(self.state[i][j][k])
                    ax.text(k, j, label, ha='center', va='center', fontsize=12, color='black')
                    ax.add_patch(matplotlib.patches.Rectangle((k-0.5, j-0.5), 1, 1, edgecolor='black', linewidth=1, facecolor=self.color_dict[self.state[i][j][k]]))
        plt.show()

    def get_face_3D_position(self,i,j,k):
        if i == 0 : #top
            return (j,k,self.size), (j,k+1,self.size), (j+1,k+1,self.size), (j,k+1,self.size)
        elif i == 5: #bottom
            return (j,k,0), (j+1,k,0), (j+1,k+1,0), (j,k+1,0)
        elif i == 1: #left
            return (0,j,k), (0,j+1,k), (0,j+1,k+1), (0,j,k+1)
        elif i == 3: #right
            return (self.size,j,k), (self.size,j+1,k), (self.size,j+1,k+1), (self.size,j,k+1)
        elif i == 2: #front
            return (j,0,k), (j+1,0,k), (j+1,0,k+1), (j,0,k+1)
        elif i == 4: #back 
            return (j,self.size,k), (j+1,self.size,k), (j+1,self.size,k+1), (j,self.size,k+1)

    def show_cube_3D(self):
        """Show the current state of the cube, with color and borders, for visualization."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Draw faces
        for i in range(6):
            for j in range(self.size):
                for k in range(self.size):
                    v1, v2, v3, v4 = self.get_face_3D_position(i, j, k)
                    ax.add_collection3d(Poly3DCollection([[v1, v2, v3, v4]], color=self.color_dict[self.state[i][j][k]]))

        # Draw edges
        for i in range(6):
            for j in range(self.size):
                for k in range(self.size):
                    v1, v2, v3, v4 = self.get_face_3D_position(i, j, k)
                    edges = [ [v1, v2], [v2, v3], [v3, v4], [v4, v1] ]
                    ax.add_collection3d(Line3DCollection(edges, color='black', linewidths=1, linestyles='solid'))

        ax.set_xlim([0, self.size])
        ax.set_ylim([0, self.size])
        ax.set_zlim([0, self.size])
        ax.set_axis_off()
        ax.set_aspect('equal', adjustable='box')       
        plt.show()
        
cube_variable = Cube(3)
cube_variable.show_cube_2D()
cube_variable.vertical_twist(1,1)
cube_variable.show_cube_2D()
cube_variable.horizontal_twist(0,1)
cube_variable.show_cube_2D()
