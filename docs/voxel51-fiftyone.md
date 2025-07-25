<!-- prettier-ignore -->
<div align="center">
<p align="center">

<!-- prettier-ignore -->
<img src="https://user-images.githubusercontent.com/25985824/106288517-2422e000-6216-11eb-871d-26ad2e7b1e59.png" height="55px"> &nbsp;
<img src="https://user-images.githubusercontent.com/25985824/106288518-24bb7680-6216-11eb-8f10-60052c519586.png" height="50px">

</p>
</div>

# FiftyOne: The Open-Source Tool for Computer Vision and Data Curation

**Supercharge your computer vision projects with FiftyOne, the open-source tool that helps you build high-quality datasets and improve your models!**  ([Back to Top](#top))

[FiftyOne](https://github.com/voxel51/fiftyone) is an open-source tool designed to accelerate the development of computer vision models by providing a powerful platform for data exploration, model analysis, and dataset management.  It allows you to visualize, analyze, and improve your datasets and models with ease.

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

*   **Visualize and Explore:** Easily visualize and explore images, videos, and their associated labels using an interactive interface.
*   **Interactive Embeddings Panel:** Explore data through embeddings to select points of interest and view corresponding samples/labels.
*   **Model Evaluation & Analysis:** Evaluate model performance, identify failure modes, and fine-tune models for improved accuracy.
*   **Advanced Data Curation:** Quickly find and fix data issues, annotation errors, and edge cases to enhance data quality.
*   **Rich Integrations:** Seamlessly integrate with popular deep learning libraries, including PyTorch, Hugging Face, and Ultralytics.
*   **Open and Extensible:** Customize and extend FiftyOne with plugins to fit your specific needs and workflows.

## Quickstart

Get started with FiftyOne in seconds:

```python
import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset("quickstart")
session = fo.launch_app(dataset)
```

For a deeper dive, explore the [Quickstart Colab Notebook](https://colab.research.google.com/github/voxel51/fiftyone-examples/blob/master/examples/quickstart.ipynb).

## Installation

Install FiftyOne using pip:

```bash
pip install fiftyone
```

For detailed installation instructions, including source installations and prerequisites, see the full [README](https://github.com/voxel51/fiftyone#installation).

## Additional Resources

Explore these resources to enhance your FiftyOne experience:

| [FiftyOne Enterprise](https://voxel51.com/enterprise) | [VoxelGPT](https://github.com/voxel51/voxelgpt) | [Plugins](https://voxel51.com/plugins) | [Vector Search](https://voxel51.com/blog/the-computer-vision-interface-for-vector-search) | [Dataset Zoo](https://docs.voxel51.com/dataset_zoo/index.html) | [Model Zoo](https://docs.voxel51.com/model_zoo/index.html) | [FiftyOne Brain](https://docs.voxel51.com/brain.html) |

## Documentation

Access comprehensive documentation for FiftyOne:

| [Tutorials](https://voxel51.com/docs/fiftyone/tutorials/index.html) | [Recipes](https://voxel51.com/docs/fiftyone/recipes/index.html) | [Examples](https://github.com/voxel51/fiftyone-examples) | [User Guide](https://voxel51.com/docs/fiftyone/user_guide/index.html) | [CLI Documentation](https://voxel51.com/docs/fiftyone/cli/index.html) | [API Reference](https://voxel51.com/docs/fiftyone/api/fiftyone.html) |

## FiftyOne Enterprise

For production-grade, collaborative, cloud-native workloads, check out [FiftyOne Enterprise](https://voxel51.com/enterprise).

## FAQ & Troubleshooting

Refer to our [common issues](https://docs.voxel51.com/getting_started/troubleshooting.html) page to troubleshoot installation issues. Consult our [frequently asked questions](https://docs.voxel51.com/faq/index.html) for answers to common queries.

If you encounter an issue, please [open an issue on GitHub](https://github.com/voxel51/fiftyone/issues) or contact us on [Discord](https://discord.gg/fiftyone-community).

## Community

Join the FiftyOne community:

[![Discord](https://img.shields.io/badge/Discord-7289DA?logo=discord&logoColor=white)](https://discord.gg/fiftyone-community)
[![Medium](https://img.shields.io/badge/Medium-12100E?logo=medium&logoColor=white)](https://medium.com/voxel51)
[![Twitter](https://img.shields.io/badge/Twitter-1DA1F2?logo=twitter&logoColor=white)](https://twitter.com/voxel51)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?logo=linkedin&logoColor=white)](https://www.linkedin.com/company/voxel51)
[![Facebook](https://img.shields.io/badge/Facebook-1877F2?logo=facebook&logoColor=white)](https://www.facebook.com/voxel51)

**Share your projects and tag us on social media with @Voxel51 and #FiftyOne!**

## Contributors

FiftyOne is open source and welcomes community contributions! See the [contribution guide](https://github.com/voxel51/fiftyone/blob/develop/CONTRIBUTING.md) to get involved.

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