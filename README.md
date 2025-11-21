# -Nonsinglet-distribution-functions-using-the-neural-network-and-genetic-algorithm
This repository contains the full implementation, data, and supplementary material
for the article:

**“Nonsinglet distribution functions using the neural network and genetic algorithm”**  
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
q(x,Q^2) = \sum_{n=0}^{N_L} a_n(Q^2) L_n(-\ln x).
\]

To ensure flexibility, the input distribution at the scale \( Q_0^2 \) is modeled by
a neural network, and its parameters are optimized using a genetic algorithm
minimizing the global \(\chi^2\).

The framework supports three initial scales:

- \(Q_0^2 = 1\ \text{GeV}^2\)  
- \(Q_0^2 = 1.69\ \text{GeV}^2\)  
- \(Q_0^2 = 4\ \text{GeV}^2\)

---

## ⚙️ **Installation**

```bash
git clone(https://github.com/elhamastaraki90-lang/-Nonsinglet-distribution-functions-using-theneural-network-and-genetic-algorithm.git)
cd Nonsinglet_PDFs_NN_GA
pip install -r requirements.txt
