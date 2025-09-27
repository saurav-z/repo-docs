# Flower: Build Cutting-Edge Federated AI Systems

Flower is a powerful and flexible framework designed to make federated AI accessible to everyone, enabling collaborative machine learning while preserving data privacy.  [Explore the Flower Repository](https://github.com/adap/flower).

<p align="center">
  <a href="https://flower.ai/">
    <img src="https://flower.ai/_next/image/?url=%2F_next%2Fstatic%2Fmedia%2Fflwr-head.4d68867a.png&w=384&q=75" width="140px" alt="Flower Website" />
  </a>
</p>

<p align="center">
    <a href="https://flower.ai/">Website</a> |
    <a href="https://flower.ai/blog">Blog</a> |
    <a href="https://flower.ai/docs/">Docs</a> |
    <a href="https://flower.ai/events/flower-ai-summit-2025">Summit</a> |
    <a href="https://flower.ai/events/flower-ai-day-2025">AI Day</a> |
    <a href="https://flower.ai/join-slack">Slack</a>
    <br /><br />
</p>

[![GitHub license](https://img.shields.io/github/license/adap/flower)](https://github.com/adap/flower/blob/main/LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/adap/flower/blob/main/CONTRIBUTING.md)
![Build](https://github.com/adap/flower/actions/workflows/framework.yml/badge.svg)
[![Downloads](https://static.pepy.tech/badge/flwr)](https://pepy.tech/project/flwr)
[![Docker Hub](https://img.shields.io/badge/Docker%20Hub-flwr-blue)](https://hub.docker.com/u/flwr)
[![Slack](https://img.shields.io/badge/Chat-Slack-red)](https://flower.ai/join-slack)

## Key Features

*   **Customizable:** Adaptable to various federated learning use cases, offering a wide range of configurations.
*   **Extensible:** Built for AI research, allowing you to extend and override components.
*   **Framework-Agnostic:** Compatible with popular machine learning frameworks like PyTorch, TensorFlow, Hugging Face Transformers, and more.
*   **Understandable:** Designed with maintainability in mind, fostering community contributions and collaboration.

## Federated Learning Tutorials

Learn the fundamentals of federated learning and how to implement them with Flower.

*   [What is Federated Learning?](https://flower.ai/docs/framework/main/en/tutorial-series-what-is-federated-learning.html)
*   [An Introduction to Federated Learning](https://flower.ai/docs/framework/main/en/tutorial-series-get-started-with-flower-pytorch.html)
*   [Using Strategies in Federated Learning](https://flower.ai/docs/framework/main/en/tutorial-series-use-a-federated-learning-strategy-pytorch.html)
*   [Customize a Flower Strategy](https://flower.ai/docs/framework/main/en/tutorial-series-build-a-strategy-from-scratch-pytorch.html)
*   [Communicate Custom Messages](https://flower.ai/docs/framework/main/en/tutorial-series-customize-the-client-pytorch.html)

## 30-Minute Federated Learning Tutorial

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adap/flower/blob/main/examples/flower-in-30-minutes/tutorial.ipynb) (or open the [Jupyter Notebook](https://github.com/adap/flower/blob/main/examples/flower-in-30-minutes/tutorial.ipynb))

## Documentation

Access comprehensive documentation to get started with Flower.

*   [Installation](https://flower.ai/docs/framework/how-to-install-flower.html)
*   [Quickstart (TensorFlow)](https://flower.ai/docs/framework/tutorial-quickstart-tensorflow.html)
*   [Quickstart (PyTorch)](https://flower.ai/docs/framework/tutorial-quickstart-pytorch.html)
*   [Quickstart (Hugging Face)](https://flower.ai/docs/framework/tutorial-quickstart-huggingface.html)
*   [Quickstart (PyTorch Lightning)](https://flower.ai/docs/framework/tutorial-quickstart-pytorch-lightning.html)
*   [Quickstart (Pandas)](https://flower.ai/docs/framework/tutorial-quickstart-pandas.html)
*   [Quickstart (fastai)](https://flower.ai/docs/framework/tutorial-quickstart-fastai.html)
*   [Quickstart (JAX)](https://flower.ai/docs/framework/tutorial-quickstart-jax.html)
*   [Quickstart (scikit-learn)](https://flower.ai/docs/framework/tutorial-quickstart-scikitlearn.html)
*   [Quickstart (Android [TFLite])](https://flower.ai/docs/framework/tutorial-quickstart-android.html)
*   [Quickstart (iOS [CoreML])](https://flower.ai/docs/framework/tutorial-quickstart-ios.html)

## Flower Baselines

Explore a collection of community-contributed projects reproducing experiments from popular federated learning publications.

*   [DASHA](https://github.com/adap/flower/tree/main/baselines/dasha)
*   [DepthFL](https://github.com/adap/flower/tree/main/baselines/depthfl)
*   [FedBN](https://github.com/adap/flower/tree/main/baselines/fedbn)
*   [FedMeta](https://github.com/adap/flower/tree/main/baselines/fedmeta)
*   [FedMLB](https://github.com/adap/flower/tree/main/baselines/fedmlb)
*   [FedPer](https://github.com/adap/flower/tree/main/baselines/fedper)
*   [FedProx](https://github.com/adap/flower/tree/main/baselines/fedprox)
*   [FedNova](https://github.com/adap/flower/tree/main/baselines/fednova)
*   [HeteroFL](https://github.com/adap/flower/tree/main/baselines/heterofl)
*   [FedAvgM](https://github.com/adap/flower/tree/main/baselines/fedavgm)
*   [FedRep](https://github.com/adap/flower/tree/main/baselines/fedrep)
*   [FedStar](https://github.com/adap/flower/tree/main/baselines/fedstar)
*   [FedWav2vec2](https://github.com/adap/flower/tree/main/baselines/fedwav2vec2)
*   [FjORD](https://github.com/adap/flower/tree/main/baselines/fjord)
*   [MOON](https://github.com/adap/flower/tree/main/baselines/moon)
*   [niid-Bench](https://github.com/adap/flower/tree/main/baselines/niid_bench)
*   [TAMUNA](https://github.com/adap/flower/tree/main/baselines/tamuna)
*   [FedVSSL](https://github.com/adap/flower/tree/main/baselines/fedvssl)
*   [FedXGBoost](https://github.com/adap/flower/tree/main/baselines/hfedxgboost)
*   [FedPara](https://github.com/adap/flower/tree/main/baselines/fedpara)
*   [FedAvg](https://github.com/adap/flower/tree/main/baselines/flwr_baselines/flwr_baselines/publications/fedavg_mnist)
*   [FedOpt](https://github.com/adap/flower/tree/main/baselines/flwr_baselines/flwr_baselines/publications/adaptive_federated_optimization)

## Flower Usage Examples

Find numerous code examples showcasing Flower in various scenarios with popular machine learning frameworks.

*   [Quickstart examples](https://github.com/adap/flower/tree/main/examples)
*   [Other examples](https://github.com/adap/flower/tree/main/examples)

## Community

Join the vibrant Flower community of researchers and engineers to collaborate and contribute. [Join Slack](https://flower.ai/join-slack).

<a href="https://github.com/adap/flower/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=adap/flower&columns=10" />
</a>

## Citation

Cite Flower in your publications:

```bibtex
@article{beutel2020flower,
  title={Flower: A Friendly Federated Learning Research Framework},
  author={Beutel, Daniel J and Topal, Taner and Mathur, Akhil and Qiu, Xinchi and Fernandez-Marques, Javier and Gao, Yan and Sani, Lorenzo and Kwing, Hei Li and Parcollet, Titouan and Gusmão, Pedro PB de and Lane, Nicholas D},
  journal={arXiv preprint arXiv:2007.14390},
  year={2020}
}
```

## Contributing to Flower

Contribute to the Flower project by following the guidelines in [CONTRIBUTING.md](CONTRIBUTING.md).