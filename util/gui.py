import tkinter as tk
import sys
import numpy as np
import time

class GUI:
    def __init__(self, block_width=20, board_shape=(22,10)):

        self.root = tk.Tk()
        self.canvas = tk.Canvas(master=self.root, width=block_width*board_shape[1], height=block_width*board_shape[0])
        self.canvas.pack()
        self.board_shape = board_shape 
        self.block_width = block_width

        self.cdict = {0: 'black', 1: 'white', -1: 'grey'}
    def update_canvas(self, board):
        
        if board.shape != self.board_shape:
            return False

        bw = self.block_width
        for x in range(self.board_shape[1]):
            for y in range(self.board_shape[0]):
                coord = (x * bw, y * bw, (x+1) * bw, (y+1) * bw)
                self.canvas.create_rectangle(*coord, fill=self.cdict[board[y][x]])
        self.root.update_idletasks()


if __name__ == '__main__':
    g = GUI()
    with open('board_output', 'rb') as f:
        while True:
            s = f.read()
            if len(s) == 220:
                b = np.fromstring(s, dtype=np.int8).reshape(22, 10)
                g.update_canvas(b)
            f.seek(0)
            time.sleep(.1)
