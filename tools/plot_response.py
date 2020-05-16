from model.model_vv import Model_VV as Model
from model.model_vv import variance_bound
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')

m = Model()

m.load()

data = np.load('data/dump.npz')

state = data['states']
value = np.squeeze(data['values'])
variance = np.squeeze(data['variance'])
weight = np.squeeze(data['weights'])

chunksize = 256

results = []
for i in range(0, len(state), chunksize):
    results.append(m.inference(state[i:i+chunksize]))

value_pred = np.squeeze(np.concatenate([r[0] for r in results]))
variance_pred = np.squeeze(np.concatenate([r[1] for r in results]))

v_size = int(len(state) * 0.1)

data = [value, variance]
pred = [value_pred, variance_pred]

data_train = [d[:-v_size] for d in data]
data_valid = [d[-v_size:] for d in data]

pred_train = [d[:-v_size] for d in pred]
pred_valid = [d[-v_size:] for d in pred]

def plot_data(truth, pred, weight, bins=100, p=1, suffix=''):
    value_t, variance_t = truth
    value_p, variance_p = pred

    loss = (np.log(variance_p.clip(min=variance_bound) / variance_t.clip(min=variance_bound)) + 
           (variance_t.clip(min=variance_bound) + (value_t - value_p) ** 2) / variance_p.clip(min=variance_bound) - 1)

    value_range = (min(np.percentile(value_t, p), np.percentile(value_p, p)), 
                  max(np.percentile(value_t, 100-p), np.percentile(value_p, 100-p)))
    variance_range = (min(np.percentile(variance_t, p), np.percentile(variance_p, p)), 
                     max(np.percentile(variance_t, 100-p), np.percentile(variance_p, 100-p)))

    plt.xlabel('value truth')
    plt.ylabel('value predict')
    plt.hist2d(value_t, value_p, bins=bins, range=(value_range, value_range), norm=mpl.colors.LogNorm())
    plt.savefig('./tmp/value_tp{}.png'.format(suffix))
    plt.cla()
    plt.clf()

    plt.xlabel('variance truth')
    plt.ylabel('variance predict')
    plt.hist2d(variance_t, variance_p, bins=bins, range=(variance_range, variance_range), norm=mpl.colors.LogNorm())
    plt.savefig('./tmp/variance_tp{}.png'.format(suffix))
    plt.cla()
    plt.clf()

    weight_range = (weight.min(), weight.max()) 

    plt.xlabel('value diff (t-p)')
    plt.ylabel('weight')
    plt.hist2d(value_t - value_p, weight, bins=bins, norm=mpl.colors.LogNorm())
    plt.savefig('./tmp/value_diff_weight{}.png'.format(suffix))
    plt.cla()
    plt.clf()

    plt.xlabel('value diff (t-p)')
    plt.ylabel('loss')
    plt.hist2d(value_t - value_p, loss, bins=bins, norm=mpl.colors.LogNorm())
    plt.savefig('./tmp/value_diff_loss{}.png'.format(suffix))
    plt.cla()
    plt.clf()

    plt.xlabel('value (truth)')
    plt.ylabel('loss')
    plt.hist2d(value_t, loss, bins=bins, norm=mpl.colors.LogNorm())
    plt.savefig('./tmp/value_t_loss{}.png'.format(suffix))
    plt.cla()
    plt.clf()

    plt.xlabel('value (predict)')
    plt.ylabel('loss')
    plt.hist2d(value_p, loss, bins=bins, norm=mpl.colors.LogNorm())
    plt.savefig('./tmp/value_p_loss{}.png'.format(suffix))
    plt.cla()
    plt.clf()

    plt.xlabel('variance diff (t-p)')
    plt.ylabel('weight')
    plt.hist2d(variance_t - variance_p, weight, bins=bins, norm=mpl.colors.LogNorm())
    plt.savefig('./tmp/variance_diff_weight{}.png'.format(suffix))
    plt.cla()
    plt.clf()

    plt.xlabel('variance diff (t-p)')
    plt.ylabel('loss')
    plt.hist2d(variance_t - variance_p, loss, bins=bins, norm=mpl.colors.LogNorm())
    plt.savefig('./tmp/variance_diff_loss{}.png'.format(suffix))
    plt.cla()
    plt.clf()

    plt.xlabel('variance (truth)')
    plt.ylabel('loss')
    plt.hist2d(variance_t, loss, bins=bins, norm=mpl.colors.LogNorm())
    plt.savefig('./tmp/variance_t_loss{}.png'.format(suffix))
    plt.cla()
    plt.clf()

    plt.xlabel('variance (predict)')
    plt.ylabel('loss')
    plt.hist2d(variance_p, loss, bins=bins, norm=mpl.colors.LogNorm())
    plt.savefig('./tmp/variance_p_loss{}.png'.format(suffix))
    plt.cla()
    plt.clf()

    plt.xlabel('variance diff (t-p)')
    plt.ylabel('value diff (t-p)')
    plt.hist2d(variance_t - variance_p, value_t - value_p, bins=bins, norm=mpl.colors.LogNorm())
    plt.savefig('./tmp/vv_diff{}.png'.format(suffix))
    plt.cla()
    plt.clf()

    plt.xlabel('weight')
    plt.ylabel('loss')
    plt.hist2d(weight, loss, bins=bins, norm=mpl.colors.LogNorm())
    plt.savefig('./tmp/weight_loss{}.png'.format(suffix))
    plt.cla()
    plt.clf()

    plt.xlabel('variance (predict)')
    plt.ylabel('value (predict)')
    plt.hist2d(variance_p, value_p, bins=bins, norm=mpl.colors.LogNorm())
    plt.savefig('./tmp/vv{}.png'.format(suffix))
    plt.cla()
    plt.clf()



plot_data(data_train, pred_train, weight[:-v_size], suffix='_train')
plot_data(data_valid, pred_valid, weight[-v_size:], suffix='_valid')
