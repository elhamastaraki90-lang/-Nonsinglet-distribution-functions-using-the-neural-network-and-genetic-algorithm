# -Nonsinglet-distribution-functions-using-the-neural-network-and-genetic-algorithm
This repository contains the full implementation, data, and supplementary material
for the article:

Nonsinglet distribution functions using the neural network and genetic algorithm
_Submitted to European Physical Journal A (EPJ A)._

The project provides a hybrid numerical–analytical framework for extracting
**nonsinglet PDFs** at LO, NLO, and NNLO using:

- Mellin-space analytical DGLAP evolution  
- Laguerre polynomial reconstruction in Bjorken-\(x\)  
- Neural network parameterization  
- Genetic Algorithm global optimization  


## 🔍 **Overview of the Method**

The nonsinglet evolution equation is solved analytically in Mellin-\(N\) space:

\[
\frac{d q_{\text{NS}}(N,Q^2)}{d\ln Q^2}
 = \gamma_{\text{NS}}(N,\alpha_s) \, q_{\text{NS}}(N,Q^2),
\]

followed by inverse reconstruction using Laguerre polynomials:

\[
q(x,Q0²) = \sum_{n=0}^{N_L} a_n(Q^2) L_n(-\ln x).
\]

To ensure flexibility, the input distribution at the scale Q0² is modeled by a neural network, and its parameters are optimized using a genetic algorithm
minimizing the global \(\chi^2\).

The framework supports three initial scales:

- Q0² = 1GeV²  
- Q0² = 1.69GeV²
- Q0² = 4GeV²

---

 Purpose
This project is designed for fitting the parameters of the NNLO non-singlet DGLAP evolution model using a hybrid approach that combines a Genetic Algorithm (GA) and a three-layer Artificial Neural Network (ANN).

Using structure function data:(𝑥,Q²,𝐹2𝑝,𝐹2𝑑,𝐹2ns), alongside the kinematic variables x and 𝑄^2, the model retrieves the following parameters:𝑎𝑢,𝑏𝑢,𝑐𝑢,𝑑𝑢,𝑎𝑑,𝑏𝑑,𝑐𝑑,𝑑𝑑,Λ2

📂 Input Data
Main input file: The data used in this study correspond to the experimental measurements of the BCDMS, SLAC, NMC, H1,
and ZEUS collaborations
Columns:x, Q, Q², F2p, F2d, F2ns
The dataset represents simulated or experimental structure function values for proton, deuteron, and non-singlet channels, prepared at a chosen starting scale Q0² (default: 4 GeV²).

⚙️ Methodology
Data Preprocessing – Load CSV and split into 80% training / 20% validation.
Surrogate Modeling with ANN –
Network architecture: [64, 64, 32] neurons in hidden layers.
Activation: tanh for hidden layers, linear for output.
Optimization with GA –
Population size: 219
Mutation rate: 0.05
Crossover rate: 0.80
Early stopping: stop after 50 generations with no improvement.
Fine-Tuning – Locally improve GA solutions using the trained ANN.
Evaluation – Compute 𝑅^2 and RMSE for all channels.
Multi-Run Averaging – Perform 30 independent runs to get mean ± standard deviation.
📊 Output Files
inverse_coefficients_hybrid_per_runs.csv – Coefficients and 𝑅^2 for each run.
ga_histories.csv – Best fitness values over generations for each run.
inverse_coefficients_hybrid_results.csv – Mean and standard deviation of coefficients across runs.
🔄 Changing Q0² 
While most existing public repositories hard-code 
Q0²=4GeV^2, this script allows you to make it a configurable parameter. Adjust the initial-scale filter in load_data() to match your desired Q0² and ensure downstream model functions use it consistently.

📌 Requirements
Python ≥ 3.9
TensorFlow ≥ 2.8
NumPy ≥ 1.20
Pandas ≥ 1.3
mpmath ≥ 1.3.0 (if exact NNLO forward evolution is enabled)
⏱ Runtime
On a standard CPU, a full 30-run hybrid GA+ANN execution may take between 30–60 minutes, depending on dataset size and hardware.

🧪 Citation
If you use this code in an academic publication, please cite the experimental data sources (BCDMS, SLAC, NMC, H1, ZEUS) and acknowledge this Hybrid GA+ANN implementation for NNLO DGLAP inverse modeling.txt
 Scope of This Repository
This repository contains only the NNLO non‑singlet DGLAP inversion code.

- Q0² fixed: The starting evolution scale is hard‑coded to: Q0²=4GeV²

- **Important**: Although the accompanying paper presents results for LO, NLO and various values of Q0² (1, 1.69, 4 GeV²), those are **not included** in this public code release.

- **Reason**: The NNLO + Q0²=4 setup matches the most stable and commonly used configuration in high‑precision DIS non‑singlet fits and allows exact reproduction of the paper's NNLO results in a reproducible open‑source form.


