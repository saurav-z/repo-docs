<!-- prettier-ignore -->
<div align="center">
<p align="center">

<!-- prettier-ignore -->
<img src="https://user-images.githubusercontent.com/25985824/106288517-2422e000-6216-11eb-871d-26ad2e7b1e59.png" height="55px"> &nbsp;
<img src="https://user-images.githubusercontent.com/25985824/106288518-24bb7680-6216-11eb-8f10-60052c519586.png" height="50px">

**The open-source tool for building high-quality datasets and computer vision
models**

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

</p>
</div>

# FiftyOne: The Ultimate Toolkit for Computer Vision 👁️

**FiftyOne is an open-source tool that empowers you to build superior computer vision datasets and models with ease.** Explore your data, analyze model performance, and improve your results with FiftyOne.  Check out the [original repository](https://github.com/voxel51/fiftyone).

## Key Features

*   **Visualize Complex Datasets:** Explore images, videos, and labels with an interactive visual interface.
*   **Explore Embeddings:**  Easily understand the relationships within your data.
*   **Analyze and Improve Models:** Evaluate model performance, identify failure modes, and refine your models effectively.
*   **Advanced Data Curation:**  Quickly identify and resolve data quality issues, annotation errors, and edge cases.
*   **Rich Integrations:** Compatible with leading deep learning libraries, including PyTorch, Hugging Face, and Ultralytics.
*   **Open and Extensible:** Customize and expand FiftyOne to meet your project's specific requirements.

## Installation

```bash
pip install fiftyone
```

For detailed installation options and troubleshooting, please refer to the [installation guide](https://voxel51.com/docs/fiftyone/getting_started/install.html).

## Quickstart

Get started in seconds! Run the following in a Python shell:

```python
import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset("quickstart")
session = fo.launch_app(dataset)
```

Explore common workflows with the quickstart dataset using the [Colab notebook](https://colab.research.google.com/github/voxel51/fiftyone-examples/blob/master/examples/quickstart.ipynb).

## Additional Resources

*   [FiftyOne Enterprise](https://voxel51.com/enterprise)
*   [VoxelGPT](https://github.com/voxel51/voxelgpt)
*   [Plugins](https://voxel51.com/plugins)
*   [Vector Search](https://voxel51.com/blog/the-computer-vision-interface-for-vector-search)
*   [Dataset Zoo](https://docs.voxel51.com/dataset_zoo/index.html)
*   [Model Zoo](https://docs.voxel51.com/model_zoo/index.html)
*   [FiftyOne Brain](https://docs.voxel51.com/brain.html)

## Documentation

Comprehensive documentation is available at [fiftyone.ai](https://fiftyone.ai).

*   [Tutorials](https://voxel51.com/docs/fiftyone/tutorials/index.html)
*   [Recipes](https://voxel51.com/docs/fiftyone/recipes/index.html)
*   [Examples](https://github.com/voxel51/fiftyone-examples)
*   [User Guide](https://voxel51.com/docs/fiftyone/user_guide/index.html)
*   [CLI Documentation](https://voxel51.com/docs/fiftyone/cli/index.html)
*   [API Reference](https://voxel51.com/docs/fiftyone/api/fiftyone.html)

## FiftyOne Enterprise

Scale your visual AI projects with the cloud-native capabilities of [FiftyOne Enterprise](https://voxel51.com/enterprise), enabling secure collaboration on massive datasets and seamless integration with your compute resources.

## FAQ & Troubleshooting

Refer to the [common issues page](https://docs.voxel51.com/getting_started/troubleshooting.html) for help with installation. If you still need help, check out the [FAQ](https://docs.voxel51.com/faq/index.html) or contact us by [opening an issue on GitHub](https://github.com/voxel51/fiftyone/issues) or contact us on [Discord](https://discord.gg/fiftyone-community).

## Join Our Community

Connect with the FiftyOne community:

*   [![Discord](https://img.shields.io/badge/Discord-7289DA?logo=discord&logoColor=white)](https://discord.gg/fiftyone-community)
*   [![Medium](https://img.shields.io/badge/Medium-12100E?logo=medium&logoColor=white)](https://medium.com/voxel51)
*   [![Twitter](https://img.shields.io/badge/Twitter-1DA1F2?logo=twitter&logoColor=white)](https://twitter.com/voxel51)
*   [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?logo=linkedin&logoColor=white)](https://www.linkedin.com/company/voxel51)
*   [![Facebook](https://img.shields.io/badge/Facebook-1877F2?logo=facebook&logoColor=white)](https://www.facebook.com/voxel51)

**Share your FiftyOne journey on social media and tag us @Voxel51 and #FiftyOne!**

## Contributors

FiftyOne is an open-source project, and we welcome community contributions!  See the [contribution guide](https://github.com/voxel51/fiftyone/blob/develop/CONTRIBUTING.md) to learn how to get involved.

Special thanks to these amazing contributors:

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