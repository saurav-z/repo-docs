<div align="center">
<p align="center">

<!-- prettier-ignore -->
<img src="https://user-images.githubusercontent.com/25985824/106288517-2422e000-6216-11eb-871d-26ad2e7b1e59.png" height="55px"> &nbsp;
<img src="https://user-images.githubusercontent.com/25985824/106288518-24bb7680-6216-11eb-8f10-60052c519586.png" height="50px">

</p>
</div>

# FiftyOne: The Open-Source Toolkit for Modern Computer Vision

**FiftyOne** empowers you to build high-quality datasets and computer vision models with an intuitive and powerful interface.  [Explore the FiftyOne GitHub Repo](https://github.com/voxel51/fiftyone).

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

*   **Visualize Datasets:** Explore images, videos, and labels with an intuitive interface.
*   **Explore Embeddings:** Easily visualize and interact with embeddings to understand your data.
*   **Analyze and Improve Models:** Evaluate model performance, pinpoint failures, and refine your models.
*   **Advanced Data Curation:** Quickly find and resolve data issues and annotation errors.
*   **Rich Integrations:** Seamlessly integrates with popular deep learning libraries.
*   **Open and Extensible:** Customize and extend FiftyOne to fit your unique needs.

<div id='additional-resources'>

## Additional Resources

| [FiftyOne Enterprise](https://voxel51.com/enterprise) | [VoxelGPT](https://github.com/voxel51/voxelgpt) | [Plugins](https://voxel51.com/plugins) | [Vector Search](https://voxel51.com/blog/the-computer-vision-interface-for-vector-search) | [Dataset Zoo](https://docs.voxel51.com/dataset_zoo/index.html) | [Model Zoo](https://docs.voxel51.com/model_zoo/index.html) | [FiftyOne Brain](https://docs.voxel51.com/brain.html) |

</div>

## Installation

```shell
pip install fiftyone
```

For detailed installation instructions and system-specific setup information, please see the detailed instructions within the original README.

## Quickstart

Get started quickly with FiftyOne:

```python
import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset("quickstart")
session = fo.launch_app(dataset)
```
Check out the [Colab notebook](https://colab.research.google.com/github/voxel51/fiftyone-examples/blob/master/examples/quickstart.ipynb) to learn more.

## Documentation

Comprehensive documentation is available at [fiftyone.ai](https://fiftyone.ai).

| [Tutorials](https://voxel51.com/docs/fiftyone/tutorials/index.html) | [Recipes](https://voxel51.com/docs/fiftyone/recipes/index.html) | [Examples](https://github.com/voxel51/fiftyone-examples) | [User Guide](https://voxel51.com/docs/fiftyone/user_guide/index.html) | [CLI Documentation](https://voxel51.com/docs/fiftyone/cli/index.html) | [API Reference](https://voxel51.com/docs/fiftyone/api/fiftyone.html) |

## FiftyOne Enterprise

Scale your visual AI projects with [FiftyOne Enterprise](https://voxel51.com/enterprise), for secure collaboration and cloud-native workflows.

## FAQ & Troubleshooting

Find answers to common questions and troubleshooting tips at the [troubleshooting page](https://docs.voxel51.com/getting_started/troubleshooting.html) and the [FAQ page](https://docs.voxel51.com/faq/index.html).  For further assistance, open an issue on GitHub or contact us on [Discord](https://discord.gg/fiftyone-community).

## Join Our Community

Connect with the FiftyOne community:

[![Discord](https://img.shields.io/badge/Discord-7289DA?logo=discord&logoColor=white)](https://discord.gg/fiftyone-community)
[![Medium](https://img.shields.io/badge/Medium-12100E?logo=medium&logoColor=white)](https://medium.com/voxel51)
[![Twitter](https://img.shields.io/badge/Twitter-1DA1F2?logo=twitter&logoColor=white)](https://twitter.com/voxel51)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?logo=linkedin&logoColor=white)](https://www.linkedin.com/company/voxel51)
[![Facebook](https://img.shields.io/badge/Facebook-1877F2?logo=facebook&logoColor=white)](https://www.facebook.com/voxel51)

**Share your FiftyOne experiences using @Voxel51 and #FiftyOne on social media!**

## Contributors

FiftyOne is an open-source project.  Contributions are welcome! See the [contribution guide](https://github.com/voxel51/fiftyone/blob/develop/CONTRIBUTING.md).

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