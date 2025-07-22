<div align="center">
<p align="center">

<!-- prettier-ignore -->
<img src="https://user-images.githubusercontent.com/25985824/106288517-2422e000-6216-11eb-871d-26ad2e7b1e59.png" height="55px"> &nbsp;
<img src="https://user-images.githubusercontent.com/25985824/106288518-24bb7680-6216-11eb-8f10-60052c519586.png" height="50px">

</p>
</div>

# FiftyOne: Supercharge Your Computer Vision Workflows

**FiftyOne is the open-source tool that empowers you to build high-quality datasets and cutting-edge computer vision models.**  [Visit the GitHub Repository](https://github.com/voxel51/fiftyone)

---

*   [Website](https://voxel51.com/fiftyone)
*   [Docs](https://voxel51.com/docs/fiftyone)
*   [Try it Now](https://colab.research.google.com/github/voxel51/fiftyone-examples/blob/master/examples/quickstart.ipynb)
*   [Tutorials](https://voxel51.com/docs/fiftyone/tutorials/index.html)
*   [Examples](https://github.com/voxel51/fiftyone-examples)
*   [Blog](https://voxel51.com/blog/)
*   [Community](https://discord.gg/fiftyone-community)

[![PyPI python](https://img.shields.io/pypi/pyversions/fiftyone)](https://pypi.org/project/fiftyone)
[![PyPI version](https://badge.fury.io/py/fiftyone.svg)](https://pypi.org/project/fiftyone)
[![Downloads](https://static.pepy.tech/badge/fiftyone)](https://pepy.tech/project/fiftyone)
[![Docker Pulls](https://badgen.net/docker/pulls/voxel51/fiftyone?icon=docker&label=pulls)](https://hub.docker.com/r/voxel51/fiftyone/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Discord](https://img.shields.io/badge/Discord-7289DA?logo=discord&logoColor=white)](https://discord.gg/fiftyone-community)
[![Medium](https://img.shields.io/badge/Medium-12100E?logo=medium&logoColor=white)](https://medium.com/voxel51)
[![Mailing list](http://bit.ly/2Md9rxM)](https://share.hsforms.com/1zpJ60ggaQtOoVeBqIZdaaA2ykyk)
[![Twitter](https://img.shields.io/twitter/follow/Voxel51?style=social)](https://twitter.com/voxel51)

## Key Features

*   **Visualize and Explore:** Effortlessly visualize and explore your image and video datasets with associated labels.
*   **Interactive Embeddings:** Interactively explore embeddings, uncovering hidden patterns and relationships within your data.
*   **Model Analysis and Improvement:** Evaluate model performance, identify failure points, and refine your models for optimal results.
*   **Data Curation:** Quickly identify and correct data issues, annotation errors, and edge cases to improve dataset quality.
*   **Seamless Integrations:** Compatible with popular deep learning libraries like PyTorch, Hugging Face, and Ultralytics, ensuring a smooth workflow.
*   **Extensible and Customizable:**  Tailor FiftyOne to meet your specific requirements through its open and extensible architecture.

<div id='additional-resources'>

## Additional Resources

| [FiftyOne Enterprise](https://voxel51.com/enterprise) | [VoxelGPT](https://github.com/voxel51/voxelgpt) | [Plugins](https://voxel51.com/plugins) | [Vector Search](https://voxel51.com/blog/the-computer-vision-interface-for-vector-search) | [Dataset Zoo](https://docs.voxel51.com/dataset_zoo/index.html) | [Model Zoo](https://docs.voxel51.com/model_zoo/index.html) | [FiftyOne Brain](https://docs.voxel51.com/brain.html) |

</div>

## Installation

Install FiftyOne with a single command:

```bash
pip install fiftyone
```

For detailed installation options and prerequisites, refer to the [Installation Guide](https://voxel51.com/docs/fiftyone/getting_started/install.html).

## Quickstart

Get started in seconds! Run the following code snippet to load a sample dataset and launch the FiftyOne App:

```python
import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset("quickstart")
session = fo.launch_app(dataset)
```

Explore common workflows in the [Colab notebook](https://colab.research.google.com/github/voxel51/fiftyone-examples/blob/master/examples/quickstart.ipynb).

## Documentation

Comprehensive documentation is available at [fiftyone.ai](https://fiftyone.ai), including:

| [Tutorials](https://voxel51.com/docs/fiftyone/tutorials/index.html) | [Recipes](https://voxel51.com/docs/fiftyone/recipes/index.html) | [Examples](https://github.com/voxel51/fiftyone-examples) | [User Guide](https://voxel51.com/docs/fiftyone/user_guide/index.html) | [CLI Documentation](https://voxel51.com/docs/fiftyone/cli/index.html) | [API Reference](https://voxel51.com/docs/fiftyone/api/fiftyone.html) |

## FiftyOne Enterprise

For collaborative, cloud-native, and production-grade workflows, check out [FiftyOne Enterprise](https://voxel51.com/enterprise).

## FAQ & Troubleshooting

Find answers to common issues on the [Troubleshooting](https://docs.voxel51.com/getting_started/troubleshooting.html) page and the [FAQ](https://docs.voxel51.com/faq/index.html).  For further assistance, open an issue on [GitHub](https://github.com/voxel51/fiftyone/issues) or join the [Discord community](https://discord.gg/fiftyone-community).

## Join the Community

Connect with the FiftyOne community:

[![Discord](https://img.shields.io/badge/Discord-7289DA?logo=discord&logoColor=white)](https://discord.gg/fiftyone-community)
[![Medium](https://img.shields.io/badge/Medium-12100E?logo=medium&logoColor=white)](https://medium.com/voxel51)
[![Twitter](https://img.shields.io/badge/Twitter-1DA1F2?logo=twitter&logoColor=white)](https://twitter.com/voxel51)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?logo=linkedin&logoColor=white)](https://www.linkedin.com/company/voxel51)
[![Facebook](https://img.shields.io/badge/Facebook-1877F2?logo=facebook&logoColor=white)](https://www.facebook.com/voxel51)

🎊 **Share your FiftyOne experiences on social media with @Voxel51 and #FiftyOne!** 🎊

## Contributors

FiftyOne is an open-source project, and we welcome community contributions! Learn how to get involved in the [contribution guide](https://github.com/voxel51/fiftyone/blob/develop/CONTRIBUTING.md).

Special thanks to our contributors:

<a href="https://github.com/voxel51/fiftyone/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=voxel51/fiftyone" />
</a>

## Citation

If you use FiftyOne in your research, please cite the project:

```bibtex
@article{moore2020fiftyone,
  title={FiftyOne},
  author={Moore, B. E. and Corso, J. J.},
  journal={GitHub. Note: https://github.com/voxel51/fiftyone},
  year={2020}
}
```