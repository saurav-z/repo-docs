<!-- prettier-ignore -->
<div align="center">
<p align="center">

<!-- prettier-ignore -->
<img src="https://user-images.githubusercontent.com/25985824/106288517-2422e000-6216-11eb-871d-26ad2e7b1e59.png" height="55px"> &nbsp;
<img src="https://user-images.githubusercontent.com/25985824/106288518-24bb7680-6216-11eb-8f10-60052c519586.png" height="50px">

</p>
</div>

# FiftyOne: The Open-Source Tool for High-Quality Datasets and Computer Vision Models

**FiftyOne** is your all-in-one solution for building superior computer vision datasets and models, available on [GitHub](https://github.com/voxel51/fiftyone).

---

<div align="center">
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

</div>

## Key Features

*   **Visualize and Explore:** Interact with your datasets using a powerful and intuitive visual interface, perfect for images and videos.
*   **Interactive Embeddings:** Easily visualize and explore embeddings to identify points of interest.
*   **Model Analysis and Improvement:** Evaluate model performance, uncover failure modes, and refine your models.
*   **Advanced Data Curation:** Quickly identify and resolve data issues and edge cases.
*   **Rich Integrations:** Seamlessly integrates with popular deep learning libraries such as PyTorch, Hugging Face, and Ultralytics.
*   **Extensible and Open-Source:** Customize and expand FiftyOne with plugins to fit your specific needs.

## Installation

Install FiftyOne with a single pip command:

```shell
pip install fiftyone
```

[Installation guide](https://voxel51.com/docs/fiftyone/getting_started/install.html) for troubleshooting and other information about getting up-and-running with FiftyOne.

<details>
<summary>More installation details</summary>
### Installation options
FiftyOne supports Python 3.9 - 3.11.
For most users, we recommend installing the latest release version of FiftyOne via `pip` as shown above.
If you want to contribute to FiftyOne or install the latest development version, then you can also perform a [source install](#source-install).
See the [prerequisites section](#prerequisites) for system-specific setup information.
We strongly recommend that you install FiftyOne in a [virtual environment](https://voxel51.com/docs/fiftyone/getting_started/virtualenv.html) to maintain a clean workspace.
Consult the [installation guide](https://voxel51.com/docs/fiftyone/getting_started/install.html) for troubleshooting and other information about getting up-and-running with FiftyOne.

</details>

## Quickstart

Get started with FiftyOne immediately:

1.  Open a Python shell.
2.  Run the following code snippet:

```python
import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset("quickstart")
session = fo.launch_app(dataset)
```

3.  Explore the dataset in the FiftyOne App.
4.  Use [this Colab notebook](https://colab.research.google.com/github/voxel51/fiftyone-examples/blob/master/examples/quickstart.ipynb) for common workflows.
5. Note that if you are running the above code in a script, you must include `session.wait()` to block execution until you close the App. See [this page](https://voxel51.com/docs/fiftyone/user_guide/app.html#creating-a-session) for more information.

## Additional Resources

*   [FiftyOne Enterprise](https://voxel51.com/enterprise)
*   [VoxelGPT](https://github.com/voxel51/voxelgpt)
*   [Plugins](https://voxel51.com/plugins)
*   [Vector Search](https://voxel51.com/blog/the-computer-vision-interface-for-vector-search)
*   [Dataset Zoo](https://docs.voxel51.com/dataset_zoo/index.html)
*   [Model Zoo](https://docs.voxel51.com/model_zoo/index.html)
*   [FiftyOne Brain](https://docs.voxel51.com/brain.html)

## Documentation

Access detailed documentation for FiftyOne at [fiftyone.ai](https://fiftyone.ai):

*   [Tutorials](https://voxel51.com/docs/fiftyone/tutorials/index.html)
*   [Recipes](https://voxel51.com/docs/fiftyone/recipes/index.html)
*   [Examples](https://github.com/voxel51/fiftyone-examples)
*   [User Guide](https://voxel51.com/docs/fiftyone/user_guide/index.html)
*   [CLI Documentation](https://voxel51.com/docs/fiftyone/cli/index.html)
*   [API Reference](https://voxel51.com/docs/fiftyone/api/fiftyone.html)

## FiftyOne Enterprise

For secure collaboration on cloud-based datasets and workflows, check out [FiftyOne Enterprise](https://voxel51.com/enterprise).

## FAQ & Troubleshooting

Find solutions to common issues and answers to frequently asked questions:

*   [Common Issues](https://docs.voxel51.com/getting_started/troubleshooting.html)
*   [Frequently Asked Questions](https://docs.voxel51.com/faq/index.html)

Need further assistance?  Please [open an issue on GitHub](https://github.com/voxel51/fiftyone/issues) or contact us on [Discord](https://discord.gg/fiftyone-community).

## Join Our Community

Connect with the FiftyOne community:

[![Discord](https://img.shields.io/badge/Discord-7289DA?logo=discord&logoColor=white)](https://discord.gg/fiftyone-community)
[![Medium](https://img.shields.io/badge/Medium-12100E?logo=medium&logoColor=white)](https://medium.com/voxel51)
[![Twitter](https://img.shields.io/badge/Twitter-1DA1F2?logo=twitter&logoColor=white)](https://twitter.com/voxel51)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?logo=linkedin&logoColor=white)](https://www.linkedin.com/company/voxel51)
[![Facebook](https://img.shields.io/badge/Facebook-1877F2?logo=facebook&logoColor=white)](https://www.facebook.com/voxel51)

**Share your FiftyOne projects on social media and tag us with @Voxel51 and #FiftyOne!**

## Contributors

FiftyOne is open source and welcomes community contributions! Learn how to contribute in the [contribution guide](https://github.com/voxel51/fiftyone/blob/develop/CONTRIBUTING.md).

Special thanks to these amazing people for contributing to FiftyOne!

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