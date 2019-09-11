import sys
import threading
import time
from collections import deque
from datetime import datetime as dt
import matplotlib.pyplot as plt
from yattag import Doc
from http.server import SimpleHTTPRequestHandler, HTTPServer
sys.path.append('../')
import model.model_pytorch as M
server_address = ('', 8000)

httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)

serve = httpd.serve_forever


def parse_log(filename):

    line_cleared = []
    score = []
    training_loss = []
    validation_loss = []
    last_lines = deque(maxlen=50)
    with open(filename) as f:
        for line in f:
            substr = 'Lines Cleared:'
            substr2 = 'Score:'
            substr_tl = 'training loss:'
            substr_vl = 'validation loss:'
            idx = line.find(substr)
            idx_tl = line.find(substr_tl)
            if idx > 0:
                lc = int(line[idx+len(substr):])
                idx2 = line.find(substr2)
                sc = int(line[idx2+len(substr2):idx])
                line_cleared.append(lc)
                score.append(sc)
            elif idx_tl > 0:  
                idx_vl = line.find(substr_vl)
                tl = float(line[idx_tl+len(substr_tl):idx_vl])
                vl = float(line[idx_vl+len(substr_vl):])
                training_loss.append(tl)
                validation_loss.append(vl)
            last_lines.append(line)
    return line_cleared, score, training_loss, validation_loss, last_lines


def generate_html(log=''):
    doc, tag, text = Doc().tagtext()
    with tag('html'):
        with tag('body'):
            doc.stag('img', src='img.png')
            doc.stag('br')
            doc.stag('img', src='img_loss.png')
            doc.stag('br')
            doc.stag('img', src='conv1_weight.png')
            doc.stag('img', src='conv2_weight.png')
            doc.stag('img', src='fc1_weight.png')
            doc.stag('img', src='fc_v_weight.png')
            doc.stag('img', src='fc_var_weight.png')
            doc.stag('br')
            with doc.tag('textarea'):
                doc.attr(rows='20')
                doc.attr(cols='240')
                for l in log:
                    text(l)
            doc.stag('br')
            text('Last update:' + str(dt.now()))

    return doc.getvalue()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('No input file')
        sys.exit()
    server_thread = threading.Thread(target=serve, args=[])
    server_thread.start()

    m = M.Model(use_cuda=False)

    while True:

        filename = sys.argv[1]

        line_cleared, score, training_loss, validation_loss, last_lines = parse_log(filename)
        c1 = 'tab:blue'
        c2 = 'tab:red'
        plt.figure(figsize=(15, 7))
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

        plt.figure(figsize=(15, 7))
        plt.semilogy(training_loss, color=c1, label='Training loss')
        plt.xlabel('Iteration')
        plt.semilogy(validation_loss, color=c2, label='Validation loss')
        plt.legend()
        plt.title('Loss vs Iteration')
        plt.savefig('img_loss.png')
        plt.clf()

        plt.figure(figsize=(4, 4))
        nbins = 100
        m.load()
        plt.hist(m.model.conv1.weight.data.numpy().ravel(), bins=nbins)
        plt.title('Conv1 weights')
        plt.savefig('conv1_weight.png')
        plt.clf()
        plt.hist(m.model.conv2.weight.data.numpy().ravel(), bins=nbins)
        plt.title('Conv2 weights')
        plt.savefig('conv2_weight.png')
        plt.clf()
        plt.hist(m.model.fc1.weight.data.numpy().ravel(), bins=nbins)
        plt.title('FC1 weights')
        plt.savefig('fc1_weight.png')
        plt.clf()
        plt.hist(m.model.fc_v.weight.data.numpy().ravel(), bins=nbins)
        plt.title('FC_V weights')
        plt.savefig('fc_v_weight.png')
        plt.clf()
        plt.hist(m.model.fc_var.weight.data.numpy().ravel(), bins=nbins)
        plt.title('FC_VAR weights')
        plt.savefig('fc_var_weight.png')
        plt.clf()

        plt.close('all')
        with open('index.html', 'w') as f:
            tmp = generate_html(last_lines)

            print(tmp)
            f.write(tmp)

        time.sleep(60)
