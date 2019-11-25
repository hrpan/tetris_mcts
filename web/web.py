import sys, os, shutil, threading, time, glob, re
from importlib import reload
from collections import deque
from datetime import datetime as dt
from shutil import copy2
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from yattag import Doc
from http.server import SimpleHTTPRequestHandler, HTTPServer
sys.path.append('../')
import model.model_pytorch as M
import numpy as np
server_address = ('', 8000)

httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)

serve = httpd.serve_forever

new_log_update = True
new_model_update = True

line_cleared = []
score = []
line_cleared_per_train = []
score_per_train = []
data_accumulated = []
training_loss = []
validation_loss = []
last_lines = deque(maxlen=100)
n_rm_since_last_game = 0
m = M.Model(use_cuda=False)

def parseLog(filename):

    global line_cleared
    global score
    global line_cleared_per_train
    global score_per_train
    global data_accumulated
    global training_loss
    global validation_loss
    global last_lines
    global n_rm_since_last_game
    global new_log_update 

    last_log_update = -1
    
    score_re = 'Episode:\s*(?P<episode>\d*)\s*' \
            'Score:\s*(?P<score>\d*)\s*' \
            'Lines Cleared:\s*(?P<lines>\d*)'   
    train_re = 'Iteration:\s*(?P<iter>\d*)\s*' \
            'training loss:\s*(?P<t_loss>\d*.\d*)\s*' \
            'validation loss:\s*(?P<v_loss>\d*.\d*)\s*'
    datasize_re = 'Training data size:\s*(?P<tsize>\d*)\s*' \
            'Validation data size:\s*(?P<vsize>\d*)'
    while True:
        
        latest_log_update = os.path.getmtime(filename)
        if latest_log_update > last_log_update:
            last_log_update = latest_log_update
            line_cleared.clear()
            score.clear()
            line_cleared_per_train.clear()
            score_per_train.clear()
            data_accumulated.clear()
            training_loss.clear()
            validation_loss.clear()
            last_lines.clear()
            n_rm_since_last_game = 0
        else:
            time.sleep(1)
            continue

        with open(filename) as f:
            lc_avg_tmp = []
            sc_avg_tmp = []
            data_accum = 0
            for line in f.readlines():
                match_score_re = re.search(score_re, line)
                match_train_re = re.search(train_re, line)
                match_datasize_re = re.search(datasize_re, line)
                if match_score_re:
                    d = match_score_re.groupdict()
                    lc = int(d['lines'])
                    sc = int(d['score'])
                    line_cleared.append(lc)
                    score.append(sc)
                    lc_avg_tmp.append(lc)
                    sc_avg_tmp.append(sc)
                    data_accumulated.append(data_accum)
                    n_rm_since_last_game = 0
                elif match_train_re:
                    d = match_train_re.groupdict()
                    tl = float(d['t_loss'])
                    vl = float(d['v_loss'])
                    training_loss.append(tl)
                    validation_loss.append(vl)
                elif match_datasize_re:
                    d = match_datasize_re.groupdict()
                    tsize = int(d['tsize'])
                    vsize = int(d['vsize'])
                    data_accum += (tsize + vsize)
                if 'proceed to training' in line:
                    if lc_avg_tmp:
                        line_cleared_per_train.append((np.average(lc_avg_tmp), np.std(lc_avg_tmp)/np.sqrt(len(lc_avg_tmp))))
                        lc_avg_tmp.clear()
                    else:
                        line_cleared_per_train.append(line_cleared_per_train[-1]) 
                    if sc_avg_tmp:
                        score_per_train.append((np.average(sc_avg_tmp), np.std(sc_avg_tmp)/np.sqrt(len(sc_avg_tmp))))
                        sc_avg_tmp.clear()
                    else:
                        score_per_train.append(score_per_train[-1]) 
                if 'REMOVING UNUSED NODES' in line:
                    n_rm_since_last_game += 1
                last_lines.append(line)
            if lc_avg_tmp:
                line_cleared_per_train.append((np.average(lc_avg_tmp), np.std(lc_avg_tmp)/np.sqrt(len(lc_avg_tmp))))
            if sc_avg_tmp:
                score_per_train.append((np.average(sc_avg_tmp), np.std(sc_avg_tmp)/np.sqrt(len(sc_avg_tmp))))

        new_log_update = True


def check_model():

    global new_model_update
    global m

    latest_module_update, last_module_update = -1, -1
    latest_model_update, last_model_update = -1, -1

    while True:
        try:
            latest_module_update = os.path.getmtime('../model/model_pytorch.py')
        except:
            pass
        if latest_module_update > last_module_update:
            last_module_update = latest_module_update        
            reload(M)

        try:
            latest_model_update = os.path.getmtime('../pytorch_model/model_checkpoint')
            if latest_model_update > last_model_update:
                last_model_update = latest_model_update
                new_model_update = True

                m = M.Model(use_cuda=False)
                try:
                    copy2('../pytorch_model/model_checkpoint', './')
                    m.load(filename='./model_checkpoint')
                except:
                    pass

        except:
            time.sleep(5)

def generate_html(log='', n=None):
    doc, tag, text = Doc().tagtext()
    with tag('html'):
        with tag('body'):
            doc.stag('img', src='img.png')
            doc.stag('img', src='lc_50.png')
            doc.stag('br')
            doc.stag('img', src='lc_sc_train.png')
            doc.stag('img', src='data_accum.png')
            doc.stag('br')
            doc.stag('img', src='img_loss.png')
            doc.stag('img', src='loss_100.png')
            doc.stag('br')
            for img in sorted(glob.glob('./*_weight.png'), key=os.path.getmtime):
                doc.stag('img', src=img)
            doc.stag('br')
            with doc.tag('textarea'):
                doc.attr(rows='20')
                doc.attr(cols='240')
                for l in log:
                    text(l)
            doc.stag('br')
            if n:
                text('Number of node removals since last game: {}'.format(n))
                doc.stag('br')
            text('Last update:' + str(dt.now()))

    return doc.getvalue()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('No input file')
        sys.exit()
    server_thread = threading.Thread(target=serve, args=[])
    server_thread.start()
    
    filename = sys.argv[1]

    logger_thread = threading.Thread(target=parseLog, args=[filename])
    logger_thread.start()

    model_thread = threading.Thread(target=check_model)
    model_thread.start()

    while True:

        if new_log_update:
            c1 = 'tab:blue'
            c2 = 'tab:red'
            plt.figure(figsize=(10, 5))
            plt.plot(line_cleared, color=c1)
            plt.xlabel('Episode')
            plt.ylabel('Lines Cleared')
            plt.gca().tick_params(axis='y', labelcolor=c1)
            ax2 = plt.gca().twinx()
            ax2.plot(score, color=c2)
            ax2.set_ylabel('Score')
            ax2.tick_params(axis='y', labelcolor=c2)
            plt.title('Score / Line Clears vs Episode')
            plt.savefig('img.png')
            plt.clf()

            plt.figure(figsize=(6, 5))
            plt.hist(line_cleared[-50:], bins=10)
            plt.title('Lines Cleared in the last 50 games')
            plt.savefig('lc_50.png')
            plt.clf()

            if line_cleared_per_train and score_per_train:
                tmp, tmp2 = np.array(line_cleared_per_train), np.array(score_per_train)
                plt.figure(figsize=(8, 5))
                plt.errorbar(x=list(range(len(tmp))), y=tmp[:,0], yerr=tmp[:,1], color=c1)
                plt.xlabel('Training Sessions')
                plt.ylabel('Lines Cleared')
                plt.gca().tick_params(axis='y', labelcolor=c1)
                ax2 = plt.gca().twinx()
                ax2.errorbar(x=list(range(len(tmp2))), y=tmp2[:,0], yerr=tmp2[:,1], color=c2)
                ax2.set_ylabel('Score')
                ax2.tick_params(axis='y', labelcolor=c2)
                plt.title('Average Score / Line Clears vs Training Sessions')
                plt.savefig('lc_sc_train.png')
                plt.clf()

            plt.figure(figsize=(8, 5))
            plt.plot(data_accumulated)
            plt.title('Accumulated data vs Episode')
            plt.savefig('data_accum.png')
            plt.clf()

            plt.figure(figsize=(10, 5))
            plt.semilogy(training_loss, color=c1, label='Training loss')
            plt.xlabel('Iteration')
            plt.semilogy(validation_loss, color=c2, label='Validation loss')
            plt.legend()
            plt.title('Loss vs Iteration')
            plt.savefig('img_loss.png')
            plt.clf()

            plt.figure(figsize=(6, 5))
            plt.semilogy(training_loss[-100:], color=c1, label='Training loss')
            plt.xlabel('Iteration')
            plt.semilogy(validation_loss[-100:], color=c2, label='Validation loss')
            plt.legend()
            plt.title('Loss vs Iteration')
            plt.savefig('loss_100.png')
            plt.clf()
            plt.close('all')
        if new_model_update:
            nbins = 50
            for module in m.model.named_modules():
                if not module[0] or not hasattr(module[1], 'weight') or module[1].weight is None:
                    continue
                print(module)

                plt.figure(figsize=(4, 4))
                plt.hist(module[1].weight.data.numpy().ravel(), bins=nbins)
                plt.title('{} weights'.format(module[0]))
                plt.savefig('{}_weight.png'.format(module[0]))
                plt.clf()
            plt.close('all')

        if new_model_update or new_log_update:
            with open('index.html', 'w') as f:
                tmp = generate_html(last_lines, n_rm_since_last_game)

                #print(tmp)
                f.write(tmp)

        new_log_update = False
        new_model_update = False
        
        time.sleep(1)
