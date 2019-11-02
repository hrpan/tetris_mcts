import sdl2
import sdl2.ext
import numpy as np
import time
sdl2.ext.init()

BLACK = sdl2.ext.Color(0, 0, 0)
WHITE = sdl2.ext.Color(255, 255, 255)
GRAY = sdl2.ext.Color(128, 128, 128)

c_dict = { 0: BLACK, 1: WHITE, -1: GRAY}

class GUI:
    def __init__(self, block_size=20):
        self.window = sdl2.ext.Window('Tetris', size=(10 * block_size, 22 * block_size))
        self.window.show()
        self.wsurface = self.window.get_surface()
        self.block_size = block_size 
        
    def update_canvas(self, board):
        
        bs = self.block_size
        ws = self.wsurface
        for x in range(10):
            for y in range(22):
                coord = (x * bs, y * bs, (x+1) * bs, (y+1) * bs)
                color = c_dict[board[y][x]]
                sdl2.ext.fill(ws, color, coord)
        for x in range(10):
            sdl2.ext.line(ws, BLACK, (20 * x, 0, 20 * x, 440))
        for y in range(22):
            sdl2.ext.line(ws, BLACK, (0, 20 * y, 200, 20 * y))
                        
        self.window.refresh()
