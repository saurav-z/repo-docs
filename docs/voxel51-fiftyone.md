<div align="center">
<p align="center">

<!-- prettier-ignore -->
<img src="https://user-images.githubusercontent.com/25985824/106288517-2422e000-6216-11eb-871d-26ad2e7b1e59.png" height="55px"> &nbsp;
<img src="https://user-images.githubusercontent.com/25985824/106288518-24bb7680-6216-11eb-8f10-60052c519586.png" height="50px">

</p>
</div>

# FiftyOne: The Open-Source Tool for Computer Vision and Dataset Management

**FiftyOne** empowers you to build high-quality datasets and computer vision models with its powerful visualization, analysis, and data curation capabilities. ([Original Repo](https://github.com/voxel51/fiftyone))

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

*   **Visualize Complex Datasets:** Explore images, videos, and labels with an intuitive visual interface.
*   **Explore Embeddings:**  Visually explore high-dimensional data and identify patterns.
*   **Analyze and Improve Models:** Evaluate model performance, identify failure modes, and fine-tune your models effectively.
*   **Advanced Data Curation:** Quickly address data issues, annotation errors, and edge cases.
*   **Rich Integrations:** Seamlessly integrates with popular deep learning libraries like PyTorch, Hugging Face, and Ultralytics.
*   **Open and Extensible:** Customize and extend FiftyOne to meet your specific needs.

## Installation

```shell
pip install fiftyone
```

For detailed installation options, including source installations and prerequisites, please refer to the [installation guide](https://voxel51.com/docs/fiftyone/getting_started/install.html).

## Quickstart

Get started with FiftyOne quickly by running the following Python code:

```py
import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset("quickstart")
session = fo.launch_app(dataset)
```

Explore common workflows with the quickstart dataset by checking out this [Colab notebook](https://colab.research.google.com/github/voxel51/fiftyone-examples/blob/master/examples/quickstart.ipynb). Remember to include `session.wait()` in your script to keep the App open.

## Additional Resources

*   [FiftyOne Enterprise](https://voxel51.com/enterprise)
*   [VoxelGPT](https://github.com/voxel51/voxelgpt)
*   [Plugins](https://voxel51.com/plugins)
*   [Vector Search](https://voxel51.com/blog/the-computer-vision-interface-for-vector-search)
*   [Dataset Zoo](https://docs.voxel51.com/dataset_zoo/index.html)
*   [Model Zoo](https://docs.voxel51.com/model_zoo/index.html)
*   [FiftyOne Brain](https://docs.voxel51.com/brain.html)

## Documentation

Find comprehensive documentation to maximize the value of FiftyOne: [fiftyone.ai](https://fiftyone.ai)

*   [Tutorials](https://voxel51.com/docs/fiftyone/tutorials/index.html)
*   [Recipes](https://voxel51.com/docs/fiftyone/recipes/index.html)
*   [Examples](https://github.com/voxel51/fiftyone-examples)
*   [User Guide](https://voxel51.com/docs/fiftyone/user_guide/index.html)
*   [CLI Documentation](https://voxel51.com/docs/fiftyone/cli/index.html)
*   [API Reference](https://voxel51.com/docs/fiftyone/api/fiftyone.html)

## FiftyOne Enterprise

For scalable, collaborative, and cloud-native enterprise workloads, consider [FiftyOne Enterprise](https://voxel51.com/enterprise).

## FAQ & Troubleshooting

Visit our [common issues](https://docs.voxel51.com/getting_started/troubleshooting.html) page to resolve installation problems and the [frequently asked questions](https://docs.voxel51.com/faq/index.html) for more answers. If you are still encountering difficulties, please [open an issue on GitHub](https://github.com/voxel51/fiftyone/issues) or contact us via [Discord](https://discord.gg/fiftyone-community).

## Join Our Community

Connect with us on:

[![Discord](https://img.shields.io/badge/Discord-7289DA?logo=discord&logoColor=white)](https://discord.gg/fiftyone-community)
[![Medium](https://img.shields.io/badge/Medium-12100E?logo=medium&logoColor=white)](https://medium.com/voxel51)
[![Twitter](https://img.shields.io/badge/Twitter-1DA1F2?logo=twitter&logoColor=white)](https://twitter.com/voxel51)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?logo=linkedin&logoColor=white)](https://www.linkedin.com/company/voxel51)
[![Facebook](https://img.shields.io/badge/Facebook-1877F2?logo=facebook&logoColor=white)](https://www.facebook.com/voxel51)

**Share your FiftyOne success stories on social media with @Voxel51 and #FiftyOne!**

## Contributors

FiftyOne is an open-source project and welcomes community contributions! See the [contribution guide](https://github.com/voxel51/fiftyone/blob/develop/CONTRIBUTING.md) for details on how to contribute.

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