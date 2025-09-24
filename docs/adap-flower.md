# Flower: Build Federated AI Systems with Ease

Flower is a powerful and flexible framework for building federated AI systems, enabling secure and collaborative machine learning. ([Original Repository](https://github.com/adap/flower))

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

*   **Customizable:** Tailor federated learning workflows to fit diverse use cases with flexible configuration options.
*   **Extendable:** Built for research, Flower allows for the extension and overriding of components for cutting-edge system development.
*   **Framework-Agnostic:** Integrate Flower seamlessly with your preferred machine learning frameworks like PyTorch, TensorFlow, Hugging Face Transformers, and more.
*   **Understandable:** The codebase is designed for clarity, making it easy to read, contribute to, and maintain.

## Tutorials and Resources

*   **Federated Learning Tutorial Series:** A step-by-step guide to mastering federated learning with Flower:
    *   [What is Federated Learning?](https://flower.ai/docs/framework/main/en/tutorial-series-what-is-federated-learning.html)
    *   [An Introduction to Federated Learning](https://flower.ai/docs/framework/main/en/tutorial-series-get-started-with-flower-pytorch.html)
    *   [Using Strategies in Federated Learning](https://flower.ai/docs/framework/main/en/tutorial-series-use-a-federated-learning-strategy-pytorch.html)
    *   [Customize a Flower Strategy](https://flower.ai/docs/framework/main/en/tutorial-series-build-a-strategy-from-scratch-pytorch.html)
    *   [Communicate Custom Messages](https://flower.ai/docs/framework/main/en/tutorial-series-customize-the-client-pytorch.html)
*   **30-Minute Federated Learning Tutorial:** Get started quickly with this interactive tutorial: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adap/flower/blob/main/examples/flower-in-30-minutes/tutorial.ipynb) (or open the [Jupyter Notebook](https://github.com/adap/flower/blob/main/examples/flower-in-30-minutes/tutorial.ipynb))
*   **Documentation:** Comprehensive documentation to guide you through the Flower ecosystem: [Flower Docs](https://flower.ai/docs/)
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

Explore community-contributed projects that reproduce experiments from popular federated learning publications. Contribute your own work to expand the baselines!

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

Explore a range of code examples demonstrating Flower's capabilities with various machine learning frameworks:

*   [Quickstart examples](https://github.com/adap/flower/tree/main/examples):
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

*   [Other examples](https://github.com/adap/flower/tree/main/examples):
    *   [Raspberry Pi & Nvidia Jetson Tutorial](https://github.com/adap/flower/tree/main/examples/embedded-devices)
    *   [PyTorch: From Centralized to Federated](https://github.com/adap/flower/tree/main/examples/pytorch-from-centralized-to-federated)
    *   [Vertical FL](https://github.com/adap/flower/tree/main/examples/vertical-fl)
    *   [Federated Finetuning of OpenAI's Whisper](https://github.com/adap/flower/tree/main/examples/whisper-federated-finetuning)
    *   [Federated Finetuning of Large Language Model](https://github.com/adap/flower/tree/main/examples/flowertune-llm)
    *   [Federated Finetuning of a Vision Transformer](https://github.com/adap/flower/tree/main/examples/flowertune-vit)
    *   [Advanced Flower with TensorFlow/Keras](https://github.com/adap/flower/tree/main/examples/advanced-tensorflow)
    *   [Advanced Flower with PyTorch](https://github.com/adap/flower/tree/main/examples/advanced-pytorch)
    *   [Comprehensive Flower+XGBoost](https://github.com/adap/flower/tree/main/examples/xgboost-comprehensive)
    *   [Flower through Docker Compose and with Grafana dashboard](https://github.com/adap/flower/tree/main/examples/flower-via-docker-compose)
    *   [Flower with KaplanMeierFitter from the lifelines library](https://github.com/adap/flower/tree/main/examples/federated-kaplan-meier-fitter)
    *   [Sample Level Privacy with Opacus](https://github.com/adap/flower/tree/main/examples/opacus)
    *   [Sample Level Privacy with TensorFlow-Privacy](https://github.com/adap/flower/tree/main/examples/tensorflow-privacy)
    *   [Flower with a Tabular Dataset](https://github.com/adap/flower/tree/main/examples/fl-tabular)

## Community

Join the vibrant Flower community and connect with researchers and engineers.

*   [Join Slack](https://flower.ai/join-slack)
*   [Contributors](https://github.com/adap/flower/graphs/contributors)

## Citation

If you use Flower in your research, please cite it using the following BibTeX entry:

```bibtex
@article{beutel2020flower,
  title={Flower: A Friendly Federated Learning Research Framework},
  author={Beutel, Daniel J and Topal, Taner and Mathur, Akhil and Qiu, Xinchi and Fernandez-Marques, Javier and Gao, Yan and Sani, Lorenzo and Kwing, Hei Li and Parcollet, Titouan and Gusmão, Pedro PB de and Lane, Nicholas D},
  journal={arXiv preprint arXiv:2007.14390},
  year={2020}
}
```

## Contributing

We welcome contributions to Flower!  See [CONTRIBUTING.md](CONTRIBUTING.md) for details.