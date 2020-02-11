import sys
import os
import numpy as np
import argparse
import glob
import time
sys.path.append('.')
from util.Data import DataLoader
from tkinter import *

width = 450
height = 450

n_actions = 7
"""
ARG PARSE
"""
parser = argparse.ArgumentParser()

parser.add_argument('--backend', default='pytorch',help='DL backend')
parser.add_argument('--data_paths', default=[],nargs='*',help='Data paths')
parser.add_argument('--inference', default=False, help='Real time inference', action='store_true')
parser.add_argument('--update_interval', default=10, type=int, help='Update interval (ms)')
args = parser.parse_args()

backend = args.backend
data_paths = args.data_paths
inference = args.inference
update_interval = args.update_interval

"""
MODEL INIT
"""
sys.path.append('./model')
if inference:
    if backend == 'pytorch':

        from model_pytorch import Model

        m = Model()
        m.load()


def drawBoard(board, canvas):
    c_w = canvas.winfo_width()
    c_h = canvas.winfo_height()
    h, w = board.shape
    b_w = c_w // w
    b_h = c_h // h
    canvas.delete('all')
    for i in range(h):
        for j in range(w):
            _v = board[i][j]
            if _v == 0:
                color = 'black'
            elif _v == 1:
                color = 'white'
            else:
                color = 'gray'
            canvas.create_rectangle(j * b_w, i * b_h, (j+1) * b_w, (i+1) * b_h, fill=color)

def drawPolicy(policy, canvas, offset_x=10, offset_y=100):
    c_w = canvas.winfo_width()
    c_h = canvas.winfo_height()
    b_w = c_w // n_actions-1
    b_h = c_h // 2
    canvas.delete('all')
    for i, v in enumerate(policy):
        x_i = offset_x + i * b_w
        x_f = offset_x + ( i + 1 ) * b_w
        y_i = offset_y
        y_f = offset_y + b_h
        color = 'gray' + str(int(100*v))
        canvas.create_rectangle(x_i, y_i, x_f, y_f, fill=color)

index = 0

if __name__ == '__main__':
    master = Tk()
    master.geometry('%dx%d'%(width, height))
    master.resizable(False, False)
    master.title('Replay')

    list_of_data = []
    for path in data_paths:
        list_of_data += glob.glob(path)
    data = DataLoader(data_paths)

    canvas_frame = Frame(master)
    canvas_frame.grid(row=0,column=0,rowspan=10,columnspan=5)

    canvas_frame_2 = Frame(master)
    canvas_frame_2.grid(row=5,column=5,rowspan=5,columnspan=5)

    info_frame = Frame(master)
    info_frame.grid(row=0,column=5,rowspan=3,columnspan=5)

    control_frame = Frame(master)
    control_frame.grid(row=3,column=5,rowspan=1,columnspan=5)

    control_frame_2 = Frame(master)
    control_frame_2.grid(row=4,column=5,rowspan=1,columnspan=5)

    list_of_updates = []

    board_canvas = Canvas(canvas_frame,width=width//2,height=height)
    board_canvas.grid(row=1,column=1)
    def update_board_canvas(index):
        global data
        board = data.getBoard(index)
        drawBoard(board,board_canvas)
    list_of_updates.append(update_board_canvas)

    policy_canvas_label = Label(canvas_frame_2,text='Policy MCTS')
    policy_canvas_label.grid(row=0,column=0)
    policy_canvas = Canvas(canvas_frame_2,width=width//2,height=height//7)
    policy_canvas.grid(row=1,column=0)
    policy_canvas_label_2 = Label(canvas_frame_2,text='Policy prediction')
    policy_canvas_label_2.grid(row=2,column=0)
    policy_canvas_2 = Canvas(canvas_frame_2,width=width//2,height=height//7)
    policy_canvas_2.grid(row=3,column=0)
    value_label = Label(canvas_frame_2)
    value_label.grid(row=4,column=0)
    class_label = Label(canvas_frame_2)
    class_label.grid(row=5,column=0)
    def update_policy_canvas(index):
        global data
        policy = data.getPolicy(index)
        if inference:
            if backend == 'pytorch':
                pred = m.inference(data.getBoard(index)[None,None,:,:])
                value_pred = pred[0][0][0]
                policy_pred = pred[1][0]
                class_pred = 0
        else:
            value_pred = -1
            policy_pred = np.empty((n_actions,)) 
            class_pred = 0
        drawPolicy(policy,policy_canvas,offset_x=0,offset_y=0)
        drawPolicy(policy_pred,policy_canvas_2,offset_x=0,offset_y=0)
        value_label.config(text='Value prediction: %.3f'%value_pred)
        class_label.config(text='Class prediction: %d'%class_pred)
    list_of_updates.append(update_policy_canvas)

    current_index_label = Label(info_frame)
    current_index_label.pack()
    current_cycle_label = Label(info_frame)
    current_cycle_label.pack()
    current_score_label = Label(info_frame)
    current_score_label.pack()
    current_lines_label = Label(info_frame)
    current_lines_label.pack()
    current_combo_label = Label(info_frame)
    current_combo_label.pack()
    def update_info_frame(index):
        global data
        current_index_label.config(text='Current Index: %10d'%index)
        current_cycle_label.config(text='Current Cycle: %10d'%data.getCycle(index))
        current_score_label.config(text='Current Score: %10d'%data.getScore(index))
        current_lines_label.config(text='Current Lines: %10d'%data.getLines(index))
        current_combo_label.config(text='Current Combo: %10d'%data.getCombo(index))
    list_of_updates.append(update_info_frame)

    def next_index():
        global index
        index = data.bound_index(index+1)
    next_index_button = Button(control_frame,text='Next',command=next_index)
    next_index_button.grid(row=0,column=0)

    def prev_index():
        global index
        index = data.bound_index(index-1) 
    prev_index_button = Button(control_frame,text='Prev',command=prev_index)
    prev_index_button.grid(row=0,column=1)

    play_after_id = None
    def play():
        global index
        global play_after_id
        index = data.bound_index(index+1)
        play_after_id = play_button.after(update_interval,play)
    play_button = Button(control_frame,text='Play',command=play)
    play_button.grid(row=0,column=2)

    def stop():
        global play_after_id
        if play_after_id:
            master.after_cancel(play_after_id)
            play_after_id = None
    stop_button = Button(control_frame,text='Stop',command=stop)
    stop_button.grid(row=0,column=3)


    index_entry_label = Label(control_frame_2,text='Goto index:')
    index_entry_label.grid(row=0,column=0)
    def set_index_entry(e):
        global index
        index = data.bound_index(int(index_entry.get()))
        print(index)
    index_entry = Entry(control_frame_2,width=10)
    index_entry.bind("<Return>",set_index_entry)
    index_entry.grid(row=0,column=1)
    
    def global_updater():
        global index
        for u in list_of_updates:
            u(index)
        master.after(update_interval, global_updater)        
    master.after(update_interval, global_updater)        
    mainloop()
