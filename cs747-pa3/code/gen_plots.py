import subprocess
import numpy as np
from matplotlib import pyplot as plt

def save_plot_combined(values, title, xtitle, ytitle, horizon=[100, 400, 1600, 6400, 25600, 102400]):

    for algo_name in values.keys():
        print(algo_name, values[algo_name])
        plt.plot(horizon, values[algo_name],  label=algo_name)

    # plt.xscale("log")
    plt.legend()
    # plt.xticks(horizon, horizon)
    plt.title(title)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.savefig('{}.png'.format(title))
    plt.close()

def save_plot(y, x, title, xtitle, ytitle):
    plt.plot(x, y)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.savefig('{}.png'.format(title))
    plt.close()


def perfom_exp_single(algorithm, king_moves, stochastic, seed_range):
    for seed in seed_range:
        if king_moves:
            subprocess.call('python run.py --king_moves --eps 0.1 --alpha 0.5 --horizon 8000 --algorithm {} --seed {}'.format(algorithm, str(seed)),
                            shell=True)

        elif stochastic:
            subprocess.call('python run.py --stochastic --eps 0.1 --alpha 0.5 --horizon 8000 --algorithm {} --seed {}'.format(algorithm, str(seed)),
                            shell=True)
        else:
            subprocess.call('python run.py --eps 0.1 --alpha 0.5 --horizon 8000 --algorithm {} --seed {}'.format(algorithm, str(seed)),
                            shell=True)

    # make plot
    if king_moves:
        file_name = 'experiment_king_{}'.format(algorithm)
    elif stochastic:
        file_name = 'experiment_stochastic_{}'.format(algorithm)
    else:
        file_name = 'experiment_{}'.format(algorithm)

    avg_vector = np.squeeze(np.zeros((8000, 1)))
    for seed in seed_range:
        current_file = '{}_{}.txt'.format(file_name, str(seed))

        with open(current_file, 'r') as f:
            data = f.readlines()
            data = np.array([int(x) for x in data])
            avg_vector += data

    avg_vector = avg_vector / len(seed_range)
    save_plot(avg_vector, list(range(8000)), file_name, 'Time Steps', 'Num Episodes')


def perfom_exp_combined(seed_range):

    for algo_name in ['sarsa', 'ql', 'esarsa']:
        for seed in seed_range:
            subprocess.call(
                'python run.py --eps 0.1 --alpha 0.5 --horizon 8000 --algorithm {} --seed {}'.format(algo_name,
                                                                                                     str(seed)),
                shell=True)

    vals = dict()
    for algorithm in ['sarsa', 'ql', 'esarsa']:
        avg_vector = np.squeeze(np.zeros((8000, 1)))
        file_name = 'experiment_{}'.format(algorithm)
        for seed in seed_range:
            current_file = '{}_{}.txt'.format(file_name, str(seed))

            with open(current_file, 'r') as f:
                data = f.readlines()
                data = np.array([int(x) for x in data])
                avg_vector += data

        avg_vector = avg_vector / len(seed_range)
        vals[algorithm] = avg_vector

    save_plot_combined(vals, 'comparison', 'Time Steps', 'Num Episodes', list(range(8000)))

if __name__ == '__main__':
    # perfom_exp_single('sarsa', False, True, list(range(10)))
    perfom_exp_combined(list(range(10)))
    # perfom_exp_single('ql', False, False, list(range(10)))