# Flower: Build Federated AI Systems with Ease

Flower is a powerful and flexible framework that simplifies the development and deployment of federated AI applications. ([Original Repository](https://github.com/adap/flower))

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

*   **Highly Customizable:** Tailor federated learning systems to your unique needs with a wide range of configuration options.
*   **Extensible for Research:** Built with AI research in mind, allowing you to extend and override components to create cutting-edge systems.
*   **Framework-Agnostic:** Works seamlessly with popular machine learning frameworks like PyTorch, TensorFlow, Hugging Face Transformers, and many more.
*   **Easy to Understand:**  The codebase is designed for maintainability and encourages community contributions.

## Tutorials

*   **Federated Learning Tutorial Series**: A comprehensive series covering the fundamentals of federated learning and implementing them in Flower.
    *   [What is Federated Learning?](https://flower.ai/docs/framework/main/en/tutorial-series-what-is-federated-learning.html)
    *   [An Introduction to Federated Learning](https://flower.ai/docs/framework/main/en/tutorial-series-get-started-with-flower-pytorch.html)
    *   [Using Strategies in Federated Learning](https://flower.ai/docs/framework/main/en/tutorial-series-use-a-federated-learning-strategy-pytorch.html)
    *   [Customize a Flower Strategy](https://flower.ai/docs/framework/main/en/tutorial-series-build-a-strategy-from-scratch-pytorch.html)
    *   [Communicate Custom Messages](https://flower.ai/docs/framework/main/en/tutorial-series-customize-the-client-pytorch.html)

*   **30-Minute Federated Learning Tutorial:** Get started quickly with a hands-on tutorial.
    *   [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adap/flower/blob/main/examples/flower-in-30-minutes/tutorial.ipynb)

## Documentation

Explore comprehensive documentation for easy implementation of Federated Learning using Flower.

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

Flower Baselines provides community-contributed projects to reproduce experiments from popular federated learning publications, enabling researchers to quickly evaluate new ideas.

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

*   [Flower Baselines Documentation](https://flower.ai/docs/baselines/)
    *   [How to use Flower Baselines](https://flower.ai/docs/baselines/how-to-use-baselines.html)
    *   [How to contribute a new Flower Baseline](https://flower.ai/docs/baselines/how-to-contribute-baselines.html)

## Flower Usage Examples

Explore diverse code examples showcasing Flower's versatility with popular machine learning frameworks.

*   [Quickstart (TensorFlow)](https://github.com/adap/flower/tree/main/examples/quickstart-tensorflow)
*   [Quickstart (PyTorch)](https://github.com/adap/flower/tree/main/examples/quickstart-pytorch)
*   [Quickstart (Hugging Face)](https://github.com/adap/flower/tree/main/examples/quickstart-huggingface)
*   [Quickstart (PyTorch Lightning)](https://github.com/adap/flower/tree/main/examples/quickstart-pytorch-lightning)
*   [Quickstart (fastai)](https://github.com/adap/flower/tree/main/examples/quickstart-fastai)
*   [Quickstart (Pandas)](https://github.com/adap/flower/tree/main/examples/quickstart-pandas)
*   [Quickstart (JAX)](https://github.com/adap/flower/tree/main/examples/quickstart-jax)
*   [Quickstart (MONAI)](https://github.com/adap/flower/tree/main/examples/quickstart-monai)
*   [Quickstart (scikit-learn)](https://github.com/adap/flower/tree/main/examples/sklearn-logreg-mnist)
*   [Quickstart (Android [TFLite])](https://github.com/adap/flower/tree/main/examples/android)
*   [Quickstart (iOS [CoreML])](https://github.com/adap/flower/tree/main/examples/ios)
*   [Quickstart (MLX)](https://github.com/adap/flower/tree/main/examples/quickstart-mlx)
*   [Quickstart (XGBoost)](https://github.com/adap/flower/tree/main/examples/xgboost-quickstart)
*   [Quickstart (CatBoost)](https://github.com/adap/flower/tree/main/examples/quickstart-catboost)

*   [More Examples](https://github.com/adap/flower/tree/main/examples)

## Community

Join the vibrant Flower community of researchers and engineers to collaborate and contribute.

*   [Join Slack](https://flower.ai/join-slack)

<a href="https://github.com/adap/flower/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=adap/flower&columns=10" />
</a>

## Citation

If you use Flower in your research, please cite the following:

```bibtex
@article{beutel2020flower,
  title={Flower: A Friendly Federated Learning Research Framework},
  author={Beutel, Daniel J and Topal, Taner and Mathur, Akhil and Qiu, Xinchi and Fernandez-Marques, Javier and Gao, Yan and Sani, Lorenzo and Kwing, Hei Li and Parcollet, Titouan and Gusmão, Pedro PB de and Lane, Nicholas D},
  journal={arXiv preprint arXiv:2007.14390},
  year={2020}
}
```

## Contributing to Flower

Contribute and shape the future of federated AI by exploring the [CONTRIBUTING.md](CONTRIBUTING.md) to get started.