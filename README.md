# Experiments and Evaluations for `SAIUnit`

This repository contains code and experiments for evaluating the [SAIUnit](https://github.com/chaobrain/saiunit) framework. The experiments focus on training and visualizing neural network models.



## Installation

To install the required packages, run the following command:

```bash
pip install -r requirements.txt
```


## fig3-diffrax and fig4-pde-simulation

For the `Fig3-diffrax` and `fig4-pde-simulation` experiment, we should first install the [unit-aware diffrax](https://github.com/chaoming0625/diffrax) first, please run the following command:

```bash
pip install git+https://github.com/chaoming0625/diffrax.git

python xxxx.py
```





## Citation

If you use this code in your research, please consider citing the following paper:

```bibtex
@article {Wang2024brainunit,
     author = {Wang, Chaoming and He, Sichao and Luo, Shouwei and Huan, Yuxiang and Wu, Si},
     title = {BrainUnit: Integrating Physical Units into High-Performance AI-Driven Scientific Computing},
     elocation-id = {2024.09.20.614111},
     year = {2024},
     doi = {10.1101/2024.09.20.614111},
     publisher = {Cold Spring Harbor Laboratory},
     abstract = {Artificial intelligence (AI) is revolutionizing scientific research across various disciplines. The foundation of scientific research lies in rigorous scientific computing based on standardized physical units. However, current mainstream high-performance numerical computing libraries for AI generally lack native support for physical units, significantly impeding the integration of AI methodologies into scientific research. To fill this gap, we introduce BrainUnit, a unit system designed to seamlessly integrate physical units into AI libraries, with a focus on compatibility with JAX. BrainUnit offers a comprehensive library of over 2000 physical units and more than 300 unit-aware mathematical functions. It is fully compatible with JAX transformations, allowing for automatic differentiation, just-in-time compilation, vectorization, and parallelization while maintaining unit consistency. We demonstrate BrainUnit{\textquoteright}s efficacy through several use cases in brain dynamics modeling, including detailed biophysical neuron simulations, multiscale brain network modeling, neuronal activity fitting, and cognitive task training. Our results show that BrainUnit enhances the accuracy, reliability, and interpretability of scientific computations across scales, from ion channels to whole-brain networks, without significantly impacting performance. By bridging the gap between abstract computational frameworks and physical units, BrainUnit represents a crucial step towards more robust and physically grounded AI-driven scientific computing.Competing Interest StatementThe authors have declared no competing interest.},
     URL = {https://www.biorxiv.org/content/early/2024/09/22/2024.09.20.614111},
     eprint = {https://www.biorxiv.org/content/early/2024/09/22/2024.09.20.614111.full.pdf},
     journal = {bioRxiv}
}

```


