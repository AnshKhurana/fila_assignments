import subprocess
import numpy as np
import os
import  argparse

parser = argparse.ArgumentParser()

parser.add_argument('--exp', type=str)
parser.add_argument('--instance', type=str)
parser.add_argument('--algorithm', type=str)

def perform_task3():

    instance_dir = '../instances'
    for instance in os.listdir(instance_dir):
        instance_file = os.path.join(instance_dir, instance)
        for eps in [0.1, 0.5, 0.8]:
            obtained_reg = []
            for seed in range(50):
                current_out = subprocess.check_output("python bandit.py --instance {}"
                                                      " --algorithm epsilon-greedy --epsilon {}"
                                                      " --randomSeed {} --horizon 102400".format(instance_file, eps, seed), shell=True)
                reg = float(current_out.strip().split()[-1])
                obtained_reg.append(reg)
                # print("Obtained regret: ", reg)
            print("For instance {}, eps = {} : Average regret is {}".format(instance_file, eps, np.mean(obtained_reg)))


def perform_task1():
    instance_dir = '../instances'
    for instance in os.listdir(instance_dir):
        instance_file = os.path.join(instance_dir, instance)
        for algorithm in ['kl-ucb']:
            subprocess.Popen('python get_results.py  --instance {} --algorithm {} --epsilon 0.02'.format(instance_file, algorithm), shell=True)


if __name__ == '__main__':
    exit()
    args = parser.parse_args()
    experiment_name = args.exp

    if experiment_name == 't1':
        perform_task1()
    elif experiment_name == 't3':
        perform_task3()
    else:
        raise NotImplementedError(experiment_name)
