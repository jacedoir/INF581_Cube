from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Cube:
    def __init__(self, size, live_display=False):
        self.size = size
        self.state = np.chararray((6, size, size), unicode=True)
        self.position_dict = {0: "left", 1:"top", 2:"front", 3:"bottom", 4:"back", 5:"right"}
        self.color_dict = {"r": "red", "g": "green", "b": "blue", "o": "orange", "y": "yellow", "w": "black", "":"none"}
        self.live_display = live_display
        if self.live_display:
            self.figure, self.ax = plt.subplots()
            self.ax.set_xlim(-1, 3*self.size)
            self.ax.set_ylim(-3*self.size, 1)
            self.ax.axis('off')
        self.reset_state()

    def reset_state(self):
        self.state[0] = np.array([["o"]*self.size]*self.size)
        self.state[1] = np.array([["w"]*self.size]*self.size)
        self.state[2] = np.array([["g"]*self.size]*self.size)
        self.state[3] = np.array([["y"]*self.size]*self.size)
        self.state[4] = np.array([["r"]*self.size]*self.size)
        self.state[5] = np.array([["b"]*self.size]*self.size)

    def print_2D(self):
        if not(self.live_display):
            plt.figure()
        #Creation d'une matrice du patron du cube
        pattern = np.chararray((4*self.size, 3*self.size), unicode=True)
        pattern[:self.size, self.size:2*self.size] = self.state[1]
        pattern[self.size:2*self.size, :self.size] = self.state[0]
        pattern[self.size:2*self.size, self.size:2*self.size] = self.state[2]
        pattern[2*self.size:3*self.size, self.size:2*self.size] = self.state[3]
        pattern[3*self.size:, self.size:2*self.size] = np.rot90(self.state[4], 2)
        pattern[self.size:2*self.size, 2*self.size:] = self.state[5]
        #Affichage du patron du cube sous forme de carré coloré en utilisant les couleurs contenues dans pattern et color_dict
        for i in range(4*self.size):
            for j in range(3*self.size):
                if self.live_display:
                    self.figure.text(j, -i, pattern[i, j], color=self.color_dict[pattern[i, j]], ha='center', va='center')
                else:
                    plt.text(j, -i, pattern[i, j], color=self.color_dict[pattern[i, j]], ha='center', va='center')
        
        if self.live_display:
            self.figure.show()
            plt.pause(3)  # Pause pendant 3 secondes
            plt.close(self.figure)  # Fermer la figure après la pause
        else:
            plt.xlim(-1, 3*self.size)
            plt.ylim(-4*self.size, 1)
            plt.axis('off')
            plt.show()

    def horizontale_rotation(self, row, direction):
        new_state = np.copy(self.state)
        if row > self.size-1:
            raise ValueError("Row number must be between 0 and ", self.size-1)
        if direction == 1: #To the right
            new_state[2, row, :] = self.state[0, row, :]
            new_state[5, row, :] = self.state[2, row, :]
            new_state[4, row, :] = self.state[5, row, :]
            new_state[0, row, :] = self.state[4, row, :]
            if row == 0:
                new_state[1] = np.rot90(self.state[1])
            elif row == self.size-1:
                new_state[3] = np.rot90(self.state[3],3)
        elif direction == -1: #To the left
            new_state[0, row, :] = self.state[2, row, :]
            new_state[2, row, :] = self.state[5, row, :]
            new_state[5, row, :] = self.state[4, row, :]
            new_state[4, row, :] = self.state[0, row, :]
            if row == 0:
                new_state[1] = np.rot90(self.state[1],3)
            elif row == self.size-1:
                new_state[3] = np.rot90(self.state[3])
        self.state = new_state

    def verticale_rotation(self, column, direction):
        new_state = np.copy(self.state)
        if column > self.size-1:
            raise ValueError("Column number must be between 0 and ", self.size-1)
        if direction == 1: #to the top
            new_state[2, :, column] = self.state[3, :, column]
            new_state[3, :, column] = self.state[4, :, column]
            new_state[4, :, column] = self.state[1, :, column]
            new_state[1, :, column] = self.state[2, :, column]
            if column == 0:
                new_state[0] = np.rot90(self.state[0])
            elif column == self.size-1:
                new_state[5] = np.rot90(self.state[5],3)
        elif direction == -1: #to the bottom
            new_state[3, :, column] = self.state[2, :, column]
            new_state[4, :, column] = self.state[3, :, column]
            new_state[1, :, column] = self.state[4, :, column]
            new_state[2, :, column] = self.state[1, :, column]
            if column == 0:
                new_state[0] = np.rot90(self.state[0],3)
            elif column == self.size-1:
                new_state[5] = np.rot90(self.state[5])
        self.state = new_state
    
    def face_rotation(self, face, direction):
        new_state = np.copy(self.state)
        if face > self.size-1:
            raise ValueError("Face number must be between 0 and ", self.size-1)
        if direction == 1: #clockwise
            new_state[1,self.size-1-face,:] = self.state[0,:,self.size-1-face]
            new_state[5, :, face] = self.state[1,self.size-1-face,:]
            new_state[3,face,:] = self.state[5,:,self.size-1-face]
            new_state[0,:,self.size-1-face] = self.state[3,face,:]
            if face == 0:
                new_state[2] = np.rot90(self.state[2],3)
            elif face == self.size-1:
                new_state[4] = np.rot90(self.state[4])
        elif direction == -1: #counterclockwise
            new_state[0,:,self.size-1-face] = self.state[1,self.size-1-face,:]
            new_state[1,self.size-1-face,:] = self.state[5, :, self.size-1-face]
            new_state[5, :, face] = self.state[3,face,:]
            new_state[3,face,:] = self.state[0,:,self.size-1-face]
            if face == 0:
                new_state[2] = np.rot90(self.state[2])
            elif face == self.size-1:
                new_state[4] = np.rot90(self.state[4],3)
        self.state = new_state

    def shuffle(self, n):
        for i in range(n):
            face = np.random.randint(self.size)
            direction = np.random.choice([-1,1])
            operation = np.random.randint(3)
            if operation == 0:
                self.horizontale_rotation(face, direction)
            elif operation == 1:
                self.verticale_rotation(face, direction)
            else:
                self.face_rotation(face, direction)



cube = Cube(2, False)
cube.print_2D()
cube.face_rotation(1, 1)
cube.print_2D()