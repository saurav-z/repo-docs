<!-- prettier-ignore -->
<div align="center">
<p align="center">

<!-- prettier-ignore -->
<img src="https://user-images.githubusercontent.com/25985824/106288517-2422e000-6216-11eb-871d-26ad2e7b1e59.png" height="55px"> &nbsp;
<img src="https://user-images.githubusercontent.com/25985824/106288518-24bb7680-6216-11eb-8f10-60052c519586.png" height="50px">

</p>
</div>

# FiftyOne: The Open-Source Toolkit for Computer Vision Data

**FiftyOne is the leading open-source tool for building high-quality datasets and computer vision models, empowering you to explore, curate, and understand your data.** [Explore the repo](https://github.com/voxel51/fiftyone).

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

*   **Visualize Complex Datasets:** Easily explore images, videos, and associated labels in a powerful visual interface.
*   **Explore Embeddings:** Select points of interest and view the corresponding samples/labels.
*   **Analyze and Improve Models:** Evaluate model performance, identify failure modes, and fine-tune your models.
*   **Advanced Data Curation:** Quickly find and fix data issues, annotation errors, and edge cases.
*   **Rich Integrations:** Works with popular deep learning libraries like PyTorch, Hugging Face, Ultralytics, and more.
*   **Open and Extensible:** Customize and extend FiftyOne to fit your specific needs.

## Quickstart

Get started with FiftyOne in seconds!  Run the following in a Python shell:

```python
import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset("quickstart")
session = fo.launch_app(dataset)
```

Then check out
[this Colab notebook](https://colab.research.google.com/github/voxel51/fiftyone-examples/blob/master/examples/quickstart.ipynb)
to see some common workflows on the quickstart dataset.

## Installation

Install FiftyOne with a single command:

```bash
pip install fiftyone
```

For detailed installation instructions and troubleshooting, refer to the [installation guide](https://voxel51.com/docs/fiftyone/getting_started/install.html).

## Additional Resources

Explore these resources to learn more about FiftyOne:

*   [FiftyOne Enterprise](https://voxel51.com/enterprise)
*   [VoxelGPT](https://github.com/voxel51/voxelgpt)
*   [Plugins](https://voxel51.com/plugins)
*   [Vector Search](https://voxel51.com/blog/the-computer-vision-interface-for-vector-search)
*   [Dataset Zoo](https://docs.voxel51.com/dataset_zoo/index.html)
*   [Model Zoo](https://docs.voxel51.com/model_zoo/index.html)
*   [FiftyOne Brain](https://docs.voxel51.com/brain.html)

## Documentation

Find comprehensive documentation to guide your journey with FiftyOne:

*   [Tutorials](https://voxel51.com/docs/fiftyone/tutorials/index.html)
*   [Recipes](https://voxel51.com/docs/fiftyone/recipes/index.html)
*   [Examples](https://github.com/voxel51/fiftyone-examples)
*   [User Guide](https://voxel51.com/docs/fiftyone/user_guide/index.html)
*   [CLI Documentation](https://voxel51.com/docs/fiftyone/cli/index.html)
*   [API Reference](https://voxel51.com/docs/fiftyone/api/fiftyone.html)

## FiftyOne Enterprise

For production-grade, collaborative, cloud-native enterprise workloads, check out [FiftyOne Enterprise](https://voxel51.com/enterprise) 🚀

## FAQ & Troubleshooting

Refer to our [common issues](https://docs.voxel51.com/getting_started/troubleshooting.html) page for troubleshooting installation and usage issues.  Also check our [frequently asked questions](https://docs.voxel51.com/faq/index.html) for additional answers.

If you encounter an issue that the above resources don't help you resolve, feel
free to [open an issue on GitHub](https://github.com/voxel51/fiftyone/issues)
or contact us on [Discord](https://discord.gg/fiftyone-community).

## Join Our Community

Connect with us on these platforms:

[![Discord](https://img.shields.io/badge/Discord-7289DA?logo=discord&logoColor=white)](https://discord.gg/fiftyone-community)
[![Medium](https://img.shields.io/badge/Medium-12100E?logo=medium&logoColor=white)](https://medium.com/voxel51)
[![Twitter](https://img.shields.io/badge/Twitter-1DA1F2?logo=twitter&logoColor=white)](https://twitter.com/voxel51)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?logo=linkedin&logoColor=white)](https://www.linkedin.com/company/voxel51)
[![Facebook](https://img.shields.io/badge/Facebook-1877F2?logo=facebook&logoColor=white)](https://www.facebook.com/voxel51)

## Contributors

FiftyOne is an open-source project, and contributions are welcome! See the [contribution guide](https://github.com/voxel51/fiftyone/blob/develop/CONTRIBUTING.md) to learn how to get involved.

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