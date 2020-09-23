import argparse
import os
import subprocess

parser = argparse.ArgumentParser()

parser.add_argument('--instance', help='where in is a path to the instance file', type=str)
parser.add_argument('--algorithm', help='where al is one of epsilon-greedy, ucb, kl-ucb,'
                                        ' thompson-sampling, and thompson-sampling-with-hint', type=str)
parser.add_argument('--epsilon',  help='ep, where ep is a number in [0, 1].', type=float)


def get_results(instance_name, algorithm_name, eps):
    with open('output_{}_{}.txt'.format(instance_name.split('/')[-1], algorithm_name), 'w') as f:
        for horizon in [100, 400, 1600, 6400, 25600, 102400]:
            for seed in range(50):
                current_out = subprocess.check_output("python bandit.py --instance {}"
                                                      " --algorithm {} --epsilon {}"
                                                      " --randomSeed {} --horizon {}".format(instance_name, algorithm_name,
                                                                                             eps,
                                                                                             seed, horizon), shell=True)
                # current_out = current_out.decode('utf-8').strip()
                # print(type(current_out))
                print(current_out.decode('utf-8'))
                f.write(current_out.decode('utf-8'))


if __name__ == '__main__':
    args = parser.parse_args()
    get_results(args.instance, args.algorithm, args.epsilon)
