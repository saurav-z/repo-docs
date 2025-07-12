<!-- prettier-ignore -->
<div align="center">
<p align="center">

<!-- prettier-ignore -->
<img src="https://user-images.githubusercontent.com/25985824/106288517-2422e000-6216-11eb-871d-26ad2e7b1e59.png" height="55px"> &nbsp;
<img src="https://user-images.githubusercontent.com/25985824/106288518-24bb7680-6216-11eb-8f10-60052c519586.png" height="50px">
</p>
</div>

# FiftyOne: The Open-Source Toolkit for Computer Vision

**FiftyOne empowers you to build high-quality datasets and computer vision models with ease.** Access the [original repo](https://github.com/voxel51/fiftyone) for more information.

---

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

*   **Visualize Datasets:** Explore images, videos, and labels with an interactive interface.
*   **Explore Embeddings:**  Discover insights through embedding visualizations.
*   **Analyze and Improve Models:** Evaluate performance, identify failure modes, and optimize models.
*   **Advanced Data Curation:** Quickly find and fix issues in your datasets.
*   **Rich Integrations:** Seamlessly integrate with popular deep learning libraries.
*   **Open and Extensible:** Customize and extend FiftyOne to fit your specific needs.

## Installation

```bash
pip install fiftyone
```

For detailed installation instructions and prerequisites, please refer to the [installation guide](https://voxel51.com/docs/fiftyone/getting_started/install.html) in the documentation.

## Quickstart

Get started in seconds with the following code:

```python
import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset("quickstart")
session = fo.launch_app(dataset)
```

Explore the [Quickstart Colab notebook](https://colab.research.google.com/github/voxel51/fiftyone-examples/blob/master/examples/quickstart.ipynb) to see common workflows.

## Additional Resources

| Resource                                                     | Link                                                                                                        |
| ------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------- |
| FiftyOne Enterprise                                            | [FiftyOne Enterprise](https://voxel51.com/enterprise)                                                         |
| VoxelGPT                                                    | [VoxelGPT](https://github.com/voxel51/voxelgpt)                                                              |
| Plugins                                                      | [Plugins](https://voxel51.com/plugins)                                                                        |
| Vector Search                                                | [Vector Search](https://voxel51.com/blog/the-computer-vision-interface-for-vector-search)                   |
| Dataset Zoo                                                  | [Dataset Zoo](https://docs.voxel51.com/dataset_zoo/index.html)                                             |
| Model Zoo                                                    | [Model Zoo](https://docs.voxel51.com/model_zoo/index.html)                                                   |
| FiftyOne Brain                                               | [FiftyOne Brain](https://docs.voxel51.com/brain.html)                                                        |

## Documentation

Access comprehensive documentation to learn more about FiftyOne:

| Resource                                                             | Link                                                                                             |
| -------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| Tutorials                                                              | [Tutorials](https://voxel51.com/docs/fiftyone/tutorials/index.html)                                |
| Recipes                                                                | [Recipes](https://voxel51.com/docs/fiftyone/recipes/index.html)                                  |
| Examples                                                               | [Examples](https://github.com/voxel51/fiftyone-examples)                                         |
| User Guide                                                             | [User Guide](https://voxel51.com/docs/fiftyone/user_guide/index.html)                             |
| CLI Documentation                                                      | [CLI Documentation](https://voxel51.com/docs/fiftyone/cli/index.html)                                |
| API Reference                                                          | [API Reference](https://voxel51.com/docs/fiftyone/api/fiftyone.html)                               |

## FiftyOne Enterprise

For collaborative, cloud-native, and production-grade visual AI workloads, explore [FiftyOne Enterprise](https://voxel51.com/enterprise).

## FAQ and Troubleshooting

Find answers to common issues and troubleshoot installation problems on the [common issues](https://docs.voxel51.com/getting_started/troubleshooting.html) page and [frequently asked questions](https://docs.voxel51.com/faq/index.html) page. Contact us on [Discord](https://discord.gg/fiftyone-community) if needed.

## Join Our Community

Connect with us and stay updated on the latest news:

*   [Discord](https://discord.gg/fiftyone-community)
*   [Medium](https://medium.com/voxel51)
*   [Twitter](https://twitter.com/voxel51)
*   [LinkedIn](https://www.linkedin.com/company/voxel51)
*   [Facebook](https://www.facebook.com/voxel51)

🎊 **Share your FiftyOne success stories on social media using @Voxel51 and #FiftyOne!** 🎊

## Contributors

FiftyOne is open source, and community contributions are welcome! Learn how to contribute in the [contribution guide](https://github.com/voxel51/fiftyone/blob/develop/CONTRIBUTING.md).

<a href="https://github.com/voxel51/fiftyone/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=voxel51/fiftyone" />
</a>

## Citation

If you use FiftyOne in your research, please cite it:

```bibtex
@article{moore2020fiftyone,
  title={FiftyOne},
  author={Moore, B. E. and Corso, J. J.},
  journal={GitHub. Note: https://github.com/voxel51/fiftyone},
  year={2020}
}
```