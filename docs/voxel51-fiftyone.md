<div align="center">
<p align="center">

<!-- prettier-ignore -->
<img src="https://user-images.githubusercontent.com/25985824/106288517-2422e000-6216-11eb-871d-26ad2e7b1e59.png" height="55px"> &nbsp;
<img src="https://user-images.githubusercontent.com/25985824/106288518-24bb7680-6216-11eb-8f10-60052c519586.png" height="50px">

**The open-source tool for building high-quality datasets and computer vision models**

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

# FiftyOne: The Open-Source Computer Vision Tool for Data-Centric AI

FiftyOne is an open-source tool designed to help you build better computer vision models by improving your datasets and model analysis. Explore and contribute to the project on [GitHub](https://github.com/voxel51/fiftyone).

## Key Features

*   **Visualize Complex Datasets:** Easily explore images, videos, and associated labels in a powerful visual interface.
    [![Visualize Datasets](https://github.com/user-attachments/assets/9dc2db88-967d-43fa-bda0-85e4d5ab6a7a)](https://docs.voxel51.com/user_guide/app.html)

*   **Explore Embeddings:** Select points of interest and view the corresponding samples/labels.
    [![Explore Embeddings](https://github.com/user-attachments/assets/246faeb7-dcab-4e01-9357-e50f6b106da7)](https://docs.voxel51.com/user_guide/app.html#embeddings-panel)

*   **Analyze and Improve Models:** Evaluate model performance, identify failure modes, and fine-tune your models.
    [![Analyze Models](https://github.com/user-attachments/assets/8c32d6c4-51e7-4fea-9a3c-2ffd9690f5d6)](https://docs.voxel51.com/user_guide/evaluation.html)

*   **Advanced Data Curation:** Quickly find and fix data issues, annotation errors, and edge cases.
    [![Data Curation](https://github.com/user-attachments/assets/24fa1960-c2dd-46ae-ae5f-d58b3b84cfe4)](https://docs.voxel51.com/brain.html)

*   **Rich Integrations:** Works with popular deep learning libraries like PyTorch, Hugging Face, Ultralytics, and more.
    [![Rich Integrations](https://github.com/user-attachments/assets/de5f25e1-a967-4362-9e04-616449e745e5)](https://docs.voxel51.com/integrations/index.html)

*   **Open and Extensible:** Customize and extend FiftyOne to fit your specific needs.
    [![Open and Extensible](https://github.com/user-attachments/assets/c7ed496d-0cf7-45d6-9853-e349f1abd6f8)](https://docs.voxel51.com/plugins/index.html)

## Installation

Install FiftyOne with a simple pip command:

```bash
pip install fiftyone
```

For more detailed instructions, including source installation and troubleshooting, refer to the [installation guide](https://voxel51.com/docs/fiftyone/getting_started/install.html).

## Quickstart

Get started with FiftyOne by downloading a sample dataset and launching the app:

```python
import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset("quickstart")
session = fo.launch_app(dataset)
```
Explore common workflows in the [Colab notebook](https://colab.research.google.com/github/voxel51/fiftyone-examples/blob/master/examples/quickstart.ipynb).

## Additional Resources

*   [FiftyOne Enterprise](https://voxel51.com/enterprise)
*   [VoxelGPT](https://github.com/voxel51/voxelgpt)
*   [Plugins](https://voxel51.com/plugins)
*   [Vector Search](https://voxel51.com/blog/the-computer-vision-interface-for-vector-search)
*   [Dataset Zoo](https://docs.voxel51.com/dataset_zoo/index.html)
*   [Model Zoo](https://docs.voxel51.com/model_zoo/index.html)
*   [FiftyOne Brain](https://docs.voxel51.com/brain.html)

## Documentation

Find comprehensive documentation at [fiftyone.ai](https://fiftyone.ai):

*   [Tutorials](https://voxel51.com/docs/fiftyone/tutorials/index.html)
*   [Recipes](https://voxel51.com/docs/fiftyone/recipes/index.html)
*   [Examples](https://github.com/voxel51/fiftyone-examples)
*   [User Guide](https://voxel51.com/docs/fiftyone/user_guide/index.html)
*   [CLI Documentation](https://voxel51.com/docs/fiftyone/cli/index.html)
*   [API Reference](https://voxel51.com/docs/fiftyone/api/fiftyone.html)

## FiftyOne Enterprise

Need to collaborate securely in the cloud and automate your workflows with your compute resources? Check out [FiftyOne Enterprise](https://voxel51.com/enterprise).

## FAQ & Troubleshooting

Refer to the [common issues](https://docs.voxel51.com/getting_started/troubleshooting.html) page to troubleshoot installation issues.  Find answers in the [frequently asked questions](https://docs.voxel51.com/faq/index.html) page.  If you still need help, open an issue on [GitHub](https://github.com/voxel51/fiftyone/issues) or contact us on [Discord](https://discord.gg/fiftyone-community).

## Join Our Community

Connect with us through:

[![Discord](https://img.shields.io/badge/Discord-7289DA?logo=discord&logoColor=white)](https://discord.gg/fiftyone-community)
[![Medium](https://img.shields.io/badge/Medium-12100E?logo=medium&logoColor=white)](https://medium.com/voxel51)
[![Twitter](https://img.shields.io/badge/Twitter-1DA1F2?logo=twitter&logoColor=white)](https://twitter.com/voxel51)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?logo=linkedin&logoColor=white)](https://www.linkedin.com/company/voxel51)
[![Facebook](https://img.shields.io/badge/Facebook-1877F2?logo=facebook&logoColor=white)](https://www.facebook.com/voxel51)

**Share your FiftyOne projects on social media with @Voxel51 and #FiftyOne!**

## Contributors

FiftyOne is an open-source project, and community contributions are welcome! See the [contribution guide](https://github.com/voxel51/fiftyone/blob/develop/CONTRIBUTING.md).

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