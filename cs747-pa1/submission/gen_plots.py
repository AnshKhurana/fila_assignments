from matplotlib import pyplot as plt
import numpy as np

def save_plot(values, title, instance_name, xtitle, ytitle, horizon=[100, 400, 1600, 6400, 25600, 102400]):

    for algo_name in values.keys():
        print(algo_name, values[algo_name])
        plt.plot(horizon, values[algo_name],  label=algo_name, marker='^')

    plt.xscale("log")
    plt.legend()
    plt.xticks(horizon, horizon)
    plt.title('{} {} REG vs T'.format(instance_name, title))

    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.savefig('{}_{}.png'.format(instance_name, title))
    plt.close()

def make_t1():
    lines_to_write = []
    for instance in ['i-1', 'i-2', 'i-3']:
        values = dict()
        for algorithm in ['epsilon-greedy', 'ucb', 'kl-ucb', 'thompson-sampling']:
            values[algorithm] = []
            with open('output_{}.txt_{}.txt'.format(instance, algorithm), 'r') as f:
                horizon_reg = dict()
                for horizon in [100, 400, 1600, 6400, 25600, 102400]:
                    horizon_reg[horizon] = []
                for line in f.readlines():
                    lines_to_write.append(line)
                    hor = int(line.split(',')[-2])
                    horizon_reg[hor].append(float(line.split(',')[-1]))
                for horizon in horizon_reg.keys():
                    values[algorithm].append(np.mean(horizon_reg[horizon]))

        save_plot(values, 't1', instance, 'Horizon', 'Average REG')

    with(open('outputDataT1.txt', 'w')) as f:
        f.writelines(lines_to_write)


def make_t2():
    lines_to_write = []
    for instance in ['i-1', 'i-2', 'i-3']:
        values = dict()
        for algorithm in ['thompson-sampling', 'thompson-sampling-with-hint']:
            values[algorithm] = []
            with open('output_{}.txt_{}.txt'.format(instance, algorithm), 'r') as f:
                horizon_reg = dict()
                for horizon in [100, 400, 1600, 6400, 25600, 102400]:
                    horizon_reg[horizon] = []
                for line in f.readlines():
                    lines_to_write.append(line)
                    hor = int(line.split(',')[-2])
                    horizon_reg[hor].append(float(line.split(',')[-1]))
                for horizon in horizon_reg.keys():
                    values[algorithm].append(np.mean(horizon_reg[horizon]))

        save_plot(values, 't2', instance, 'Horizon', 'Average REG')

    with(open('outputDataT2.txt', 'w')) as f:
        f.writelines(lines_to_write)


if __name__ == '__main__':
    make_t1()
    make_t2()
