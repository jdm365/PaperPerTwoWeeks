<h1>CRAMMING: TRAINING A LANGUAGE MODEL ON A SINGLE GPU IN ONE DAY</h1>

<h3>Authors</h3>

Jonas Geiping
University of Maryland, College Park
jgeiping@umd.edu

Tom Goldstein
University of Maryland, College Park
tomg@umd.edu


<h3>Abstract</h3>
Recent trends in language modeling have focused on increasing performance
through scaling, and have resulted in an environment where training language
models is out of reach for most researchers and practitioners. While most in the
community are asking how to push the limits of extreme computation, we ask the
opposite question: How far can we get with a single GPU in just one day?
We investigate the downstream performance achievable with a transformer-based
language model trained completely from scratch with masked language modeling
for a single day on a single consumer GPU. Aside from re-analyzing nearly all
components of the pretraining pipeline for this scenario and providing a modified
pipeline with performance close to BERT, we investigate why scaling down is
hard, and which modifications actually improve performance in this scenario.
We provide evidence that even in this constrained setting, performance closely
follows scaling laws observed in large-compute settings. Through the lens of
scaling laws, we categorize a range of recent improvements to training and
architecture and discuss their merit and practical applicability (or lack thereof)
for the limited compute setting.

<h3>arxiv link -> https://arxiv.org/abs/2212.14034</h3>
<h3>Official code -> https://github.com/JonasGeiping/cramming</h3>

<h3>Additional Notes<h3>

