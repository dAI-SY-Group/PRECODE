# PRECODE - A Generic Model Extension to Prevent Deep Gradient Leakage
This repository contains the implementation of PRECODE realized as a variational bottleneck as well as an example jupyter notebook where the PRECODE module is used.
The paper including all empirical results can be found at: https://openaccess.thecvf.com/content/WACV2022/html/Scheliga_PRECODE_-_A_Generic_Model_Extension_To_Prevent_Deep_Gradient_WACV_2022_paper.html
<p align="center">
  <img src="./attack_samples.png" alt="summary" width="500">
</p>
Neural networks are trained on CIFAR-10, CIFAR-100, and Medical MNIST. During training, gradients are used for inversion attacks aiming to reconstruct original training data. Without defense (Original Model), training data can be entirely reconstructed. Noisy gradients prevent reconstruction to some extent but reduce model performance. Our proposed PRECODE extension entirely prevents reconstruction without impacting the training process.


### Please cite as:
```
@inproceedings{scheliga2022precode,
  title={PRECODE-A Generic Model Extension to Prevent Deep Gradient Leakage},
  author={Scheliga, Daniel and M{\"a}der, Patrick and Seeland, Marco},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={1849--1858},
  year={2022}
}
```

## Abstract:
Collaborative training of neural networks leverages distributed data by exchanging gradient information between different clients. Although training data entirely resides with the clients, recent work shows that training data can be reconstructed from such exchanged gradient information. To enhance privacy, gradient perturbation techniques have been proposed. However, they come at the cost of reduced model performance, increased convergence time, or increased data demand. In this paper, we introduce PRECODE, a PRivacy EnhanCing mODulE that can be used as generic extension for arbitrary model architectures. We propose a simple yet effective realization of PRECODE using variational modeling. The stochastic sampling induced by variational modeling effectively prevents privacy leakage from gradients and in turn preserves privacy of data owners. We evaluate PRECODE using state of the art gradient inversion attacks on two different model architectures trained on three datasets. In contrast to commonly used defense mechanisms, we find that our proposed modification consistently reduces the attack success rate to 0% while having almost no negative impact on model training and final performance. As a result, PRECODE reveals a promising path towards privacy enhancing model extensions.


## Requirements:
+ pytorch
+ torchvision
+ scikit-image

You can also use [conda](https://www.anaconda.com/) to recreate our virtual environment:
```
conda env create -f environment.yaml
conda activate precode
```

## Credits:
We use the attack of Geiping et al. ([arXiv](https://arxiv.org/abs/2003.14053), [GitHub](https://github.com/JonasGeiping/invertinggradients)).

The implementation of the variational bottleneck is mainly inspired by 1Konny's implementation of a Deep Variational Information Bottleneck ([GitHub](https://github.com/1Konny/VIB-pytorch))
