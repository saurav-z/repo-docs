<div align="center">
<p align="center">

<!-- prettier-ignore -->
<img src="https://user-images.githubusercontent.com/25985824/106288517-2422e000-6216-11eb-871d-26ad2e7b1e59.png" height="55px"> &nbsp;
<img src="https://user-images.githubusercontent.com/25985824/106288518-24bb7680-6216-11eb-8f10-60052c519586.png" height="50px">
</p>
</div>

# FiftyOne: Build High-Quality Datasets and Computer Vision Models

**Supercharge your visual AI projects with FiftyOne, the open-source tool that empowers you to visualize, analyze, and improve your datasets and models.**  [Get Started](https://github.com/voxel51/fiftyone)

---

**[Website](https://voxel51.com/fiftyone) | [Docs](https://voxel51.com/docs/fiftyone) | [Try it Now](https://colab.research.google.com/github/voxel51/fiftyone-examples/blob/master/examples/quickstart.ipynb) | [Tutorials](https://voxel51.com/docs/fiftyone/tutorials/index.html) | [Examples](https://github.com/voxel51/fiftyone-examples) | [Blog](https://voxel51.com/blog/) | [Community](https://discord.gg/fiftyone-community)**

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

*   **Visualize and Explore Datasets:**  Easily explore images, videos, and associated labels using an intuitive visual interface.
*   **Interactive Embeddings:**  Select points of interest and view the corresponding samples/labels.
*   **Model Analysis and Improvement:** Evaluate model performance, identify failure modes, and fine-tune models.
*   **Advanced Data Curation:** Quickly find and fix data issues, annotation errors, and edge cases.
*   **Rich Integrations:** Works seamlessly with popular deep learning libraries like PyTorch, Hugging Face, and Ultralytics.
*   **Open and Extensible:**  Customize and extend FiftyOne to fit your specific needs through plugins.

## Installation

Install FiftyOne with a single command:

```bash
pip install fiftyone
```

For detailed installation instructions, including source installations and prerequisites, please refer to the [Installation Guide](https://voxel51.com/docs/fiftyone/getting_started/install.html).

## Quickstart

Get up and running with FiftyOne in minutes!  Run the following code in a Python shell:

```python
import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset("quickstart")
session = fo.launch_app(dataset)
```

Explore the quickstart dataset using the [Colab notebook](https://colab.research.google.com/github/voxel51/fiftyone-examples/blob/master/examples/quickstart.ipynb).

## Additional Resources

*   [FiftyOne Enterprise](https://voxel51.com/enterprise)
*   [VoxelGPT](https://github.com/voxel51/voxelgpt)
*   [Plugins](https://voxel51.com/plugins)
*   [Vector Search](https://voxel51.com/blog/the-computer-vision-interface-for-vector-search)
*   [Dataset Zoo](https://docs.voxel51.com/dataset_zoo/index.html)
*   [Model Zoo](https://docs.voxel51.com/model_zoo/index.html)
*   [FiftyOne Brain](https://docs.voxel51.com/brain.html)

## Documentation

Access comprehensive documentation to guide your journey with FiftyOne.

*   [Tutorials](https://voxel51.com/docs/fiftyone/tutorials/index.html)
*   [Recipes](https://voxel51.com/docs/fiftyone/recipes/index.html)
*   [Examples](https://github.com/voxel51/fiftyone-examples)
*   [User Guide](https://voxel51.com/docs/fiftyone/user_guide/index.html)
*   [CLI Documentation](https://voxel51.com/docs/fiftyone/cli/index.html)
*   [API Reference](https://voxel51.com/docs/fiftyone/api/fiftyone.html)

## FiftyOne Enterprise

Scale your AI projects with FiftyOne Enterprise, offering secure collaboration, cloud-native capabilities, and automated workflows. Learn more at [FiftyOne Enterprise](https://voxel51.com/enterprise).

## FAQ & Troubleshooting

Find answers to common questions and troubleshooting tips in our resources.

*   [Troubleshooting Guide](https://docs.voxel51.com/getting_started/troubleshooting.html)
*   [FAQ](https://docs.voxel51.com/faq/index.html)

If you need further assistance, open an issue on [GitHub](https://github.com/voxel51/fiftyone/issues) or contact us on [Discord](https://discord.gg/fiftyone-community).

## Join Our Community

Connect with the FiftyOne community!

[![Discord](https://img.shields.io/badge/Discord-7289DA?logo=discord&logoColor=white)](https://discord.gg/fiftyone-community)
[![Medium](https://img.shields.io/badge/Medium-12100E?logo=medium&logoColor=white)](https://medium.com/voxel51)
[![Twitter](https://img.shields.io/badge/Twitter-1DA1F2?logo=twitter&logoColor=white)](https://twitter.com/voxel51)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?logo=linkedin&logoColor=white)](https://www.linkedin.com/company/voxel51)
[![Facebook](https://img.shields.io/badge/Facebook-1877F2?logo=facebook&logoColor=white)](https://www.facebook.com/voxel51)

**Share your FiftyOne experiences on social media and tag us with @Voxel51 and #FiftyOne!**

## Contributors

FiftyOne is open source and welcomes community contributions.  See our [contribution guide](https://github.com/voxel51/fiftyone/blob/develop/CONTRIBUTING.md) to get started.

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
```

Key improvements and SEO optimizations:

*   **Clear Title and Hook:**  Uses the most relevant keywords ("FiftyOne," "build datasets," "computer vision") and presents a concise value proposition.
*   **Strategic Keyword Placement:** Integrates keywords naturally throughout the headings and content.
*   **Bulleted Key Features:**  Provides a clear, scannable overview of FiftyOne's capabilities.
*   **Well-Organized Structure:** Uses headings, subheadings, and bullet points for readability and SEO benefits.
*   **Links:** Includes links to all key resources for user convenience and improved SEO.
*   **Call to Action:**  Encourages users to get started.
*   **Concise and Informative:**  Keeps the content focused on essential information.
*   **Community Links:**  More community links for engagement.
*   **Citation Section:** Included the citation information.