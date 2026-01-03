**Learning Addition with a Small Transformer** This exercise is designed to assess how you think about **experimental design**:   
● data construction, 

● evaluation, and 

● clear reporting. 

It’s not about the highest accuracy. Even if you can’t get a model to train at all (for whatever reason, maybe colab kept kicking you off the GPU), if you discuss what may be going wrong, focus on just data generation and evaluation and skip actually training all together, it can still be an excellent report. Do not spend more than a few hours on this (really). 

You will train a **small Transformer (\< 12 layers)** to add two base-10 integers with **k digits** (e.g., k=3 means 100–999). You may use **trl** (I’ll provide [a starter template in GitHub](https://github.com/lauraruis/trl_template) that can be used for inspiration) or any library you prefer (Transformers, PyTorch Lightning, nanoGPT, etc.). You can train on a **free Colab GPU** (a T4 GPU with 16GB memory), and the full training run should not take more than **an hour or so**. 

**1\) Goal** 

Build a model that takes an input like: 

● `123 + 456 =` 

and produces: 

● `579` 

Your main job is to design: 

1\. a **training distribution** (synthetically generated) and 

2\. a **test/evaluation suite** 

that convincingly answers: *what did the model learn, and how does it generalize?* 

**2\) Constraints** 

**Model**  
● Transformer with **\< 12 layers** (e.g., 4–8 layers is totally fine). 

● Choose any tokenizer strategy you want (character-level, digit tokens, BPE, etc.), but justify it. 

● Keep things realistically trainable on Colab (small embedding/hidden sizes, batch sizes, etc.). 

**Compute** 

● You should be able to run training in **one Colab session**. 

● Target runtime: **\~60 minutes** for your main run. 

**Task** 

● Base-10 addition of two non-negative integers, typically with exactly **k digits** (you choose k). 

● You decide formatting (spaces, separators, etc.). 

**3\) What you will design** 

**A. Data generation** 

Create a synthetic dataset generator. Your writeup should specify: 

1\. **Input/output format** 

○ Example: `"a + b =","c"` 

2\. **Sampling distribution** 

○ Uniform over `a,b` ∈ `[10^(k-1), 10^k - 1]`? 

○ Or a mix? 

○ If you use any non-uniform sampling, explain why. 

3\. **Train/val/test splits** 

**B. Training run** 

Pick training parameters such as batch size, model dimensions, number of training steps, learning rate, etc. 

**C. Evaluation** 

Create an evaluation plan that includes: 

1\. **In-distribution (ID) accuracy** 

○ Same format and k as training. 

2\. **Optional generalization tests (pick one or design your own test split)** ○ **Length generalization**: train on k digits, test on k+1 or k+2 digits. ○ **Distribution shift**: train on uniform; test on skewed (e.g., many 9s, or small numbers only).  
**4\) Deliverables** 

**1\) Short report (2–4 pages)** 

PDF or Markdown is fine. Include: 

● **Task setup** (format, k, model, tokenizer) 

● **Data design** (generation, splits, diagnostics) 

● **Training details** (hyperparameters, runtime, compute) 

● **Evaluation suite** (tests \+ motivation) 

● **Results** (tables/plots) 

● **Interpretation** 

● **Discussion**: how did you design the task format; how did you design the train/test split and why; what you believe the model learned (or may learn if you didn’t get it to train); what remains uncertain; what you’d do next; why is this task hard for the model; what could improve its performance? 

**2\) Code** 

A GitHub repo or zipped folder with: 

● data generator 

● training script / notebook 

● evaluation script / notebook 

● requirements / environment notes 

**3\) A results artifact** 

● a `results.json` with metrics for each evaluation slice (if the model trained) 

**5\) Rubric** 

1\. **Clarity & rigor of the evaluation design** 

2\. **Soundness of data/split methodology** 

3\. **Interpretation** 

4\. **Reproducibility & organization** 

To state it again: high accuracy is nice, but a weaker model with an excellent evaluation story can score higher than a strong model with shallow analysis. Even if your model doesn’t learn at all, understanding why \+ good experimental design can still be a great report.

**6\) Practical guidance (you can ignore, but it may help)** 

● Pick k so the task is reasonable for a small model, but you still have enough data (k=3–5 is a good range). 

● Consider using exact match accuracy as the primary metric. 

● Think about decoding, greedy or sampling. 

● Think carefully about the training data distribution and what makes this task hard for humans. 

● Think carefully about what the model may learn based on what you train on, and what you need to hold-out to properly test generalisation. 

● If you use [the trl template](https://github.com/lauraruis/trl_template), note that you can train small transformers from scratch with supervised fine-tuning by loading for example the gpt2 architecture and randomly initialising the weights (example code provided). Note that this repo also contains code to finetune LLMs, which you don’t need to do. 

**7\) Rules** 

● You may use any public code/libraries, but **write your own data generator and evaluation logic**. 

● Cite anything you borrow. 

● Feel free to use coding agents or LLMs for questions, but do the work and the report yourself.