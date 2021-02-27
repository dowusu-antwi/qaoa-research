# qaoa-research

Reproduces [Noise-Induced Barren Plateaus (NIBPS)](https://arxiv.org/pdf/2007.14384v3.pdf) via applying QAOA to MAXCUT, with depolarization error.

Run ./main (currently set to 5 trials, est. 20 sec) to see NIBPs (gradient magnitudes reducing with increasing error probability).

To increase trials, see line 17 in main.py (e.g., try 100 trials, est. 7 min, for more siginificant NIBPS).

![Gradient v. Circuit Size (1000 trials)](images/gradient-v-circuit-size.png "1000 trials, est. 45 min")
