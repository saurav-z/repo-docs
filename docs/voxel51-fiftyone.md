<div align="center">
<p align="center">

<!-- prettier-ignore -->
<img src="https://user-images.githubusercontent.com/25985824/106288517-2422e000-6216-11eb-871d-26ad2e7b1e59.png" height="55px"> &nbsp;
<img src="https://user-images.githubusercontent.com/25985824/106288518-24bb7680-6216-11eb-8f10-60052c519586.png" height="50px">

</p>
</div>

# FiftyOne: The Open-Source Toolkit for Visual AI

**FiftyOne empowers you to build high-quality datasets and cutting-edge computer vision models with ease.**  Explore, curate, and understand your data like never before!

[<img src="https://img.shields.io/pypi/pyversions/fiftyone" alt="PyPI python" />](https://pypi.org/project/fiftyone)
[<img src="https://badge.fury.io/py/fiftyone.svg" alt="PyPI version" />](https://pypi.org/project/fiftyone)
[<img src="https://static.pepy.tech/badge/fiftyone" alt="Downloads" />](https://pepy.tech/project/fiftyone)
[<img src="https://badgen.net/docker/pulls/voxel51/fiftyone?icon=docker&label=pulls" alt="Docker Pulls" />](https://hub.docker.com/r/voxel51/fiftyone/)
[<img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License" />](LICENSE)
[<img src="https://img.shields.io/badge/Discord-7289DA?logo=discord&logoColor=white" alt="Discord" />](https://discord.gg/fiftyone-community)
[<img src="https://img.shields.io/badge/Medium-12100E?logo=medium&logoColor=white" alt="Medium" />](https://medium.com/voxel51)
[<img src="http://bit.ly/2Md9rxM" alt="Mailing list" />](https://share.hsforms.com/1zpJ60ggaQtOoVeBqIZdaaA2ykyk)
[<img src="https://img.shields.io/twitter/follow/Voxel51?style=social" alt="Twitter" />](https://twitter.com/voxel51)

👉 **[Get started with FiftyOne](https://github.com/voxel51/fiftyone)**

---

## Key Features

*   **Visualize and Explore Complex Datasets:** Easily inspect images, videos, and their annotations through an intuitive, powerful interface.

*   **Uncover Insights with Embeddings:** Interact with your data through embedding visualizations to identify patterns and anomalies.

*   **Analyze and Improve Model Performance:** Evaluate, diagnose, and refine your computer vision models with comprehensive tools for error analysis.

*   **Advanced Data Curation Capabilities:** Quickly find and correct data issues, annotation mistakes, and edge cases to improve your dataset quality.

*   **Seamless Integrations:** Works with popular deep learning frameworks like PyTorch, TensorFlow, and more.

*   **Open and Extensible:** Customize and extend FiftyOne to fit your specific needs with plugins.

## Installation

Installing FiftyOne is as simple as running the following command:

```bash
pip install fiftyone
```

For detailed installation instructions and options, refer to the [Installation section](#installation) in the original README (linked above).

## Quickstart

Get up and running in minutes with our quickstart example:

```python
import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset("quickstart")
session = fo.launch_app(dataset)
```

Explore the quickstart dataset with this [Colab notebook](https://colab.research.google.com/github/voxel51/fiftyone-examples/blob/master/examples/quickstart.ipynb).

## Additional Resources

*   **[FiftyOne Enterprise](https://voxel51.com/enterprise)**
*   **[VoxelGPT](https://github.com/voxel51/voxelgpt)**
*   **[Plugins](https://voxel51.com/plugins)**
*   **[Vector Search](https://voxel51.com/blog/the-computer-vision-interface-for-vector-search)**
*   **[Dataset Zoo](https://docs.voxel51.com/dataset_zoo/index.html)**
*   **[Model Zoo](https://docs.voxel51.com/model_zoo/index.html)**
*   **[FiftyOne Brain](https://docs.voxel51.com/brain.html)**

## Documentation

Access comprehensive documentation to guide you through FiftyOne:

*   **[Tutorials](https://voxel51.com/docs/fiftyone/tutorials/index.html)**
*   **[Recipes](https://voxel51.com/docs/fiftyone/recipes/index.html)**
*   **[Examples](https://github.com/voxel51/fiftyone-examples)**
*   **[User Guide](https://voxel51.com/docs/fiftyone/user_guide/index.html)**
*   **[CLI Documentation](https://voxel51.com/docs/fiftyone/cli/index.html)**
*   **[API Reference](https://voxel51.com/docs/fiftyone/api/fiftyone.html)**

## FiftyOne Enterprise

For production-grade, collaborative, and cloud-native workloads, explore [FiftyOne Enterprise](https://voxel51.com/enterprise).

## FAQ & Troubleshooting

Find answers to common issues and troubleshooting steps in our [FAQ](https://docs.voxel51.com/faq/index.html) and [common issues](https://docs.voxel51.com/getting_started/troubleshooting.html) pages.  If you need further assistance, open an issue on [GitHub](https://github.com/voxel51/fiftyone/issues) or join our [Discord](https://discord.gg/fiftyone-community) community.

## Join the Community

Connect with the FiftyOne community:

*   [Discord](https://discord.gg/fiftyone-community)
*   [Medium](https://medium.com/voxel51)
*   [Twitter](https://twitter.com/voxel51)
*   [LinkedIn](https://www.linkedin.com/company/voxel51)
*   [Facebook](https://www.facebook.com/voxel51)

🎊 **Share your projects and tag us with @Voxel51 and #FiftyOne** 🎊

## Contributors

FiftyOne is an open-source project; we welcome contributions from the community!
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