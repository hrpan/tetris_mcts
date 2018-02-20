import sys
import numpy as np
import pandas as pd
import argparse
import glob
import time
import tensorflow as tf
sys.path.append('./model')
from model import Model
from tkinter import *

"""
ARG PARSE
"""
parser = argparse.ArgumentParser()

parser.add_argument('--data_paths',default=['./data'],nargs='*',help='Data paths')

args = parser.parse_args()

data_paths = args.data_paths

"""
MODEL INIT
"""
sess = tf.Session()
m = Model()
m.load(sess)


class Data:
    def __init__(self,data_paths=None):
        self.data_paths = data_paths
        self.index = 0 
        self.load_data()

    def load_data(self):
        list_of_data = [f for p in self.data_paths for f in glob.glob(p+'/data*')]
        dfs = [pd.read_pickle(f) for f in list_of_data]

        self.data = pd.concat(dfs,ignore_index=True)

        self.length = self.data.shape[0]

    def bound_index(self,index):
        if index >= self.length:
            return self.length-1
        elif index < 0:
            return 0
        else:
            return index

    def getBoard(self,index):
        index = self.bound_index(index)
        return self.data['board'][index]

    def getPolicy(self,index):
        index = self.bound_index(index)
        return self.data['policy'][index]

    def getCycle(self,index):
        index = self.bound_index(index)
        return self.data['cycle'][index]

def drawBoard(board,canvas,b_pix=20):
    canvas.delete('all')
    h, w = board.shape
    for i in range(h):
        for j in range(w):
            _v = board[i][j]
            if _v == 0:
                color = 'black'
            elif _v == 1:
                color = 'white'
            else:
                color = 'gray'
            canvas.create_rectangle(j * b_pix, i * b_pix, (j+1) * b_pix, (i+1) * b_pix, fill=color)

def drawPolicy(policy,canvas,blocksize=30,offset_x=10,offset_y=100):
    canvas.delete('all')
    for i, v in enumerate(policy):
        color = 'gray' + str(int(100*v))
        canvas.create_rectangle(offset_x+i*blocksize,offset_y,offset_x+(i+1)*blocksize,offset_y+blocksize,fill=color)

index = 0
update_interval = 25

if __name__ == '__main__':
    master = Tk()
    master.title('Replay')
    data = Data(data_paths)

    canvas_frame = Frame(master)
    canvas_frame.grid(row=0,column=0,rowspan=10,columnspan=5)

    canvas_frame_2 = Frame(master)
    canvas_frame_2.grid(row=3,column=5,rowspan=7,columnspan=5)

    info_frame = Frame(master)
    info_frame.grid(row=0,column=5,rowspan=1,columnspan=5)

    control_frame = Frame(master)
    control_frame.grid(row=1,column=5,rowspan=1,columnspan=5)

    control_frame_2 = Frame(master)
    control_frame_2.grid(row=2,column=5,rowspan=1,columnspan=5)

    board_canvas = Canvas(canvas_frame,width=200,height=440)
    board_canvas.grid(row=1,column=1)
    def update_board_canvas():
        global index
        global data
        board = data.getBoard(index)
        drawBoard(board,board_canvas)
        board_canvas.after(update_interval,update_board_canvas)
    update_board_canvas()

    policy_canvas_label = Label(canvas_frame_2,text='Policy MCTS')
    policy_canvas_label.grid(row=0,column=0)
    policy_canvas = Canvas(canvas_frame_2,width=200,height=50)
    policy_canvas.grid(row=1,column=0)
    policy_canvas_label_2 = Label(canvas_frame_2,text='Policy prediction')
    policy_canvas_label_2.grid(row=2,column=0)
    policy_canvas_2 = Canvas(canvas_frame_2,width=200,height=50)
    policy_canvas_2.grid(row=3,column=0)
    value_label = Label(canvas_frame_2)
    value_label.grid(row=4,column=0)
    def update_policy_canvas():
        global index
        global data
        policy = data.getPolicy(index)
        pred = m.inference(sess,[data.getBoard(index)[:,:,None]])
        value_pred = pred[0][0]
        policy_pred = pred[1][0]
        drawPolicy(policy,policy_canvas,offset_y=0)
        drawPolicy(policy_pred,policy_canvas_2,offset_y=0)
        value_label.config(text='Value prediction: %.3f'%value_pred)
        policy_canvas.after(update_interval,update_policy_canvas)
    update_policy_canvas()

    current_index_label = Label(info_frame)
    current_index_label.pack()
    def update_current_index_label():
        global index
        current_index_label.config(text='Current index: %d'%index)
        current_index_label.after(update_interval,update_current_index_label)        
    update_current_index_label()

    current_cycle_label = Label(info_frame)
    current_cycle_label.pack()
    def update_current_cycle_label():
        global index
        global data
        _c = data.getCycle(index)
        current_cycle_label.config(text='Current cycle: %d'%_c)
        current_cycle_label.after(update_interval,update_current_cycle_label)        
    update_current_cycle_label()
   

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
    
    """
    interval_entry_label = Label(control_frame_2,text='Update interval:')
    interval_entry_label.grid(row=1,column=0)
    def set_update_interval(e):
        global update_interval
        update_interval = int(interval_entry.get())
    interval_entry = Entry(control_frame_2,width=10)
    interval_entry.bind("<Return>",set_update_interval)
    interval_entry.grid(row=1,column=1)
    """
    mainloop()
