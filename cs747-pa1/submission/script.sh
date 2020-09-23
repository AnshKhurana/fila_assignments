
for h in {100,400,1600,6400,25600,102400} ;
do
for i in {0..49}; do python bandit.py --instance ../instances/i-1.txt --algorithm kl-ucb --randomSeed $i --epsilon 0.02 --horizon $h >> output_i-1.txt_kl-ucb.txt & done ;
wait;
for i in {0..49}; do python bandit.py --instance ../instances/i-2.txt --algorithm kl-ucb --randomSeed $i --epsilon 0.02 --horizon $h >> output_i-2.txt_kl-ucb.txt & done ;
wait;
for i in {0..49}; do python bandit.py --instance ../instances/i-3.txt --algorithm kl-ucb --randomSeed $i --epsilon 0.02 --horizon $h >> output_i-3.txt_kl-ucb.txt & done ;
done
