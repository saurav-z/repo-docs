<div align="center">
<p align="center">

<!-- prettier-ignore -->
<img src="https://user-images.githubusercontent.com/25985824/106288517-2422e000-6216-11eb-871d-26ad2e7b1e59.png" height="55px"> &nbsp;
<img src="https://user-images.githubusercontent.com/25985824/106288518-24bb7680-6216-11eb-8f10-60052c519586.png" height="50px">

</p>
</div>

# FiftyOne: The Open-Source Tool for Computer Vision Data Management

**FiftyOne** empowers you to build high-quality datasets and computer vision models faster by providing a powerful platform for visualizing, analyzing, and curating your data.  Explore the [FiftyOne GitHub Repository](https://github.com/voxel51/fiftyone) to get started.

---

<!-- prettier-ignore -->
<a href="https://voxel51.com/fiftyone">Website</a> •
<a href="https://voxel51.com/docs/fiftyone">Docs</a> •
<a href="https://colab.research.google.com/github/voxel51/fiftyone-examples/blob/master/examples/quickstart.ipynb">Try it Now</a> •
<a href="https://voxel51.com/docs/fiftyone/tutorials/index.html">Tutorials</a> •
<a href="https://github.com/voxel51/fiftyone-examples">Examples</a> •
<a href="https://voxel51.com/blog/">Blog</a> •
<a href="https://discord.gg/fiftyone-community">Community</a>

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

*   **Visualize and Explore:**  Effortlessly visualize and interact with your image and video datasets, along with their labels, through a powerful and intuitive visual interface.
*   **Explore Embeddings:**  Identify key insights in your data by interacting with embeddings, selecting points of interest, and viewing the corresponding samples and labels.
*   **Model Evaluation & Improvement:**  Assess and enhance your model's performance by analyzing results, identifying failure points, and refining your models with ease.
*   **Advanced Data Curation:**  Quickly identify and rectify data quality issues, annotation errors, and edge cases, leading to higher-quality datasets.
*   **Seamless Integrations:**  Integrate effortlessly with popular deep learning libraries, including PyTorch, Hugging Face, Ultralytics, and more, streamlining your workflow.
*   **Open & Extensible:** Customize and extend FiftyOne to align with your specific needs, thanks to its open-source and extensible architecture.

## Installation

Install FiftyOne with pip:

```bash
pip install fiftyone
```

For more installation options and detailed prerequisites, see the full [installation guide](https://voxel51.com/docs/fiftyone/getting_started/install.html).

## Quickstart

Get started quickly by running this code snippet in a Python shell:

```python
import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset("quickstart")
session = fo.launch_app(dataset)
```

## Additional Resources

*   [FiftyOne Enterprise](https://voxel51.com/enterprise)
*   [VoxelGPT](https://github.com/voxel51/voxelgpt)
*   [Plugins](https://voxel51.com/plugins)
*   [Vector Search](https://voxel51.com/blog/the-computer-vision-interface-for-vector-search)
*   [Dataset Zoo](https://docs.voxel51.com/dataset_zoo/index.html)
*   [Model Zoo](https://docs.voxel51.com/model_zoo/index.html)
*   [FiftyOne Brain](https://docs.voxel51.com/brain.html)

## Documentation

Access the complete documentation for FiftyOne at [fiftyone.ai](https://fiftyone.ai).

*   [Tutorials](https://voxel51.com/docs/fiftyone/tutorials/index.html)
*   [Recipes](https://voxel51.com/docs/fiftyone/recipes/index.html)
*   [Examples](https://github.com/voxel51/fiftyone-examples)
*   [User Guide](https://voxel51.com/docs/fiftyone/user_guide/index.html)
*   [CLI Documentation](https://voxel51.com/docs/fiftyone/cli/index.html)
*   [API Reference](https://voxel51.com/docs/fiftyone/api/fiftyone.html)

## FiftyOne Enterprise

For production-grade, collaborative, cloud-native enterprise workloads, explore [FiftyOne Enterprise](https://voxel51.com/enterprise).

## FAQ & Troubleshooting

Find answers to common issues and troubleshooting steps on our [common issues page](https://docs.voxel51.com/getting_started/troubleshooting.html) and in the [FAQ](https://docs.voxel51.com/faq/index.html).  If you need further assistance, open an issue on [GitHub](https://github.com/voxel51/fiftyone/issues) or join us on [Discord](https://discord.gg/fiftyone-community).

## Join Our Community

Connect with us!

[![Discord](https://img.shields.io/badge/Discord-7289DA?logo=discord&logoColor=white)](https://discord.gg/fiftyone-community)
[![Medium](https://img.shields.io/badge/Medium-12100E?logo=medium&logoColor=white)](https://medium.com/voxel51)
[![Twitter](https://img.shields.io/badge/Twitter-1DA1F2?logo=twitter&logoColor=white)](https://twitter.com/voxel51)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?logo=linkedin&logoColor=white)](https://www.linkedin.com/company/voxel51)
[![Facebook](https://img.shields.io/badge/Facebook-1877F2?logo=facebook&logoColor=white)](https://www.facebook.com/voxel51)

🎊 Share your FiftyOne projects on social media and tag us with @Voxel51 and #FiftyOne! 🎊

## Contributors

FiftyOne is an open-source project, and community contributions are welcome!  See the [contribution guide](https://github.com/voxel51/fiftyone/blob/develop/CONTRIBUTING.md) to learn more.

<a href="https://github.com/voxel51/fiftyone/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=voxel51/fiftyone" />
</a>

## Citation

```bibtex
@article{moore2020fiftyone,
  title={FiftyOne},
  author={Moore, B. E. and Corso, J. J.},
  journal={GitHub. Note: https://github.com/voxel51/fiftyone},
  year={2020}
}
```