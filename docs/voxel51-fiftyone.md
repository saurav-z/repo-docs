<div align="center">
<p align="center">

<!-- prettier-ignore -->
<img src="https://user-images.githubusercontent.com/25985824/106288517-2422e000-6216-11eb-871d-26ad2e7b1e59.png" height="55px"> &nbsp;
<img src="https://user-images.githubusercontent.com/25985824/106288518-24bb7680-6216-11eb-8f10-60052c519586.png" height="50px">

</p>
</div>

# FiftyOne: The Open-Source Toolkit for Computer Vision 

**FiftyOne empowers you to build high-quality datasets and computer vision models with its intuitive and powerful features.** ([See the original repo](https://github.com/voxel51/fiftyone))

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

*   **Visualize and Explore Datasets:** Easily visualize images, videos, and labels in a powerful visual interface.
*   **Explore Embeddings:** Interactive embeddings visualizations to identify patterns and outliers.
*   **Analyze and Improve Models:** Evaluate model performance, identify failure modes, and fine-tune models.
*   **Advanced Data Curation:** Quickly find and fix data issues, annotation errors, and edge cases with FiftyOne Brain.
*   **Rich Integrations:** Works seamlessly with popular deep learning libraries (PyTorch, Hugging Face, Ultralytics, and more).
*   **Open and Extensible:** Customize and extend FiftyOne to fit your specific needs.

## Installation

Install FiftyOne using pip:

```bash
pip install fiftyone
```

For detailed installation instructions, troubleshooting, and source installations, refer to the [installation guide](https://voxel51.com/docs/fiftyone/getting_started/install.html).

## Quickstart

Get started by running a simple command to load a dataset and launch the FiftyOne App:

```python
import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset("quickstart")
session = fo.launch_app(dataset)
```
*   See the [Colab notebook](https://colab.research.google.com/github/voxel51/fiftyone-examples/blob/master/examples/quickstart.ipynb) for more examples.

## Additional Resources

*   **FiftyOne Enterprise:** [https://voxel51.com/enterprise](https://voxel51.com/enterprise)
*   **VoxelGPT:** [https://github.com/voxel51/voxelgpt](https://github.com/voxel51/voxelgpt)
*   **Plugins:** [https://voxel51.com/plugins](https://voxel51.com/plugins)
*   **Vector Search:** [https://voxel51.com/blog/the-computer-vision-interface-for-vector-search](https://voxel51.com/blog/the-computer-vision-interface-for-vector-search)
*   **Dataset Zoo:** [https://docs.voxel51.com/dataset_zoo/index.html](https://docs.voxel51.com/dataset_zoo/index.html)
*   **Model Zoo:** [https://docs.voxel51.com/model_zoo/index.html](https://docs.voxel51.com/model_zoo/index.html)
*   **FiftyOne Brain:** [https://docs.voxel51.com/brain.html](https://docs.voxel51.com/brain.html)

## Documentation

Find comprehensive documentation at [fiftyone.ai](https://fiftyone.ai).

*   **Tutorials:** [https://voxel51.com/docs/fiftyone/tutorials/index.html](https://voxel51.com/docs/fiftyone/tutorials/index.html)
*   **Recipes:** [https://voxel51.com/docs/fiftyone/recipes/index.html](https://voxel51.com/docs/fiftyone/recipes/index.html)
*   **Examples:** [https://github.com/voxel51/fiftyone-examples](https://github.com/voxel51/fiftyone-examples)
*   **User Guide:** [https://voxel51.com/docs/fiftyone/user_guide/index.html](https://voxel51.com/docs/fiftyone/user_guide/index.html)
*   **CLI Documentation:** [https://voxel51.com/docs/fiftyone/cli/index.html](https://voxel51.com/docs/fiftyone/cli/index.html)
*   **API Reference:** [https://voxel51.com/docs/fiftyone/api/fiftyone.html](https://voxel51.com/docs/fiftyone/api/fiftyone.html)

## FiftyOne Enterprise

For collaborative cloud-native enterprise workloads, explore [FiftyOne Enterprise](https://voxel51.com/enterprise).

## FAQ & Troubleshooting

Refer to the [common issues](https://docs.voxel51.com/getting_started/troubleshooting.html) page for troubleshooting tips.  Find answers to frequently asked questions on the [FAQ page](https://docs.voxel51.com/faq/index.html).  For further assistance, open an issue on [GitHub](https://github.com/voxel51/fiftyone/issues) or contact us on [Discord](https://discord.gg/fiftyone-community).

## Join Our Community

Connect with the FiftyOne community:

[![Discord](https://img.shields.io/badge/Discord-7289DA?logo=discord&logoColor=white)](https://discord.gg/fiftyone-community)
[![Medium](https://img.shields.io/badge/Medium-12100E?logo=medium&logoColor=white)](https://medium.com/voxel51)
[![Twitter](https://img.shields.io/badge/Twitter-1DA1F2?logo=twitter&logoColor=white)](https://twitter.com/voxel51)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?logo=linkedin&logoColor=white)](https://www.linkedin.com/company/voxel51)
[![Facebook](https://img.shields.io/badge/Facebook-1877F2?logo=facebook&logoColor=white)](https://www.facebook.com/voxel51)

## Contributors

FiftyOne is an open-source project, and contributions are welcome! See the [contribution guide](https://github.com/voxel51/fiftyone/blob/develop/CONTRIBUTING.md) for details.

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