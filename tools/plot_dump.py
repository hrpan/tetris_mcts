import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

data = np.load('data/dump.npz')

state = data['states']
value = np.squeeze(data['values'])
variance = np.squeeze(data['variance'])
weight = np.squeeze(data['weights'])

_data = [value, variance, weight]

v_size = int(len(state) * 0.1)

_data_train = [d[:-v_size] for d in _data]
_data_valid = [d[-v_size:] for d in _data]

def plot_data(data, bins=200, suffix=''):

    value, variance, weight = data

    plt.xlabel('value')
    plt.hist(value, bins=bins)
    plt.yscale('log')
    plt.savefig('./tmp/value_hist{}.png'.format(suffix))
    plt.cla()
    plt.clf()

    plt.xlabel('variance')
    plt.hist(variance, bins=bins)
    plt.yscale('log')
    plt.savefig('./tmp/variance_hist{}.png'.format(suffix))
    plt.cla()
    plt.clf()

    plt.xlabel('weight')
    plt.hist(weight, bins=bins)
    plt.yscale('log')
    plt.savefig('./tmp/weight_hist{}.png'.format(suffix))
    plt.cla()
    plt.clf()

    plt.yscale('linear')
    plt.xlabel('value')
    plt.ylabel('variance')
    plt.hist2d(value, variance, bins=bins, norm=mpl.colors.LogNorm())
    plt.savefig('./tmp/value_variance{}.png'.format(suffix))
    plt.cla()
    plt.clf()

    plt.xlabel('value')
    plt.ylabel('weight')
    plt.hist2d(value, weight, bins=bins, norm=mpl.colors.LogNorm())
    plt.savefig('./tmp/value_weight{}.png'.format(suffix))
    plt.cla()
    plt.clf()

    plt.xlabel('variance')
    plt.ylabel('weight')
    plt.hist2d(variance, weight, bins=bins, norm=mpl.colors.LogNorm())
    plt.savefig('./tmp/variance_weight{}.png'.format(suffix))
    plt.cla()
    plt.clf()

plot_data(_data)
plot_data(_data_train, suffix='_train')
plot_data(_data_valid, suffix='_valid')
