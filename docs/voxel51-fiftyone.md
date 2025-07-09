<!-- prettier-ignore -->
<div align="center">
<p align="center">

<!-- prettier-ignore -->
<img src="https://user-images.githubusercontent.com/25985824/106288517-2422e000-6216-11eb-871d-26ad2e7b1e59.png" height="55px"> &nbsp;
<img src="https://user-images.githubusercontent.com/25985824/106288518-24bb7680-6216-11eb-8f10-60052c519586.png" height="50px">

</p>
</div>

# FiftyOne: The Open-Source Toolkit for Computer Vision Data Management

**FiftyOne** is an open-source tool designed to streamline the creation and analysis of high-quality datasets and computer vision models. Check out the [original repo](https://github.com/voxel51/fiftyone) for more details.

## Key Features

*   **Visualize Complex Datasets:** Explore images, videos, and labels in an intuitive visual interface.
*   **Explore Embeddings:** Select points of interest and view corresponding samples.
*   **Analyze and Improve Models:** Evaluate model performance and refine models effectively.
*   **Advanced Data Curation:** Quickly identify and correct data issues and annotation errors.
*   **Rich Integrations:** Seamlessly integrates with popular deep learning libraries such as PyTorch, Hugging Face, and Ultralytics.
*   **Open and Extensible:** Customize and extend FiftyOne to meet your unique project needs.

## Installation

```bash
pip install fiftyone
```

For detailed installation instructions, including prerequisites and source installations, refer to the [Installation section in the original README](https://github.com/voxel51/fiftyone#installation).

## Quickstart

Get started quickly by running the following Python code to download a sample dataset and launch the FiftyOne App:

```python
import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset("quickstart")
session = fo.launch_app(dataset)
```

Explore common workflows using the quickstart dataset in this [Colab notebook](https://colab.research.google.com/github/voxel51/fiftyone-examples/blob/master/examples/quickstart.ipynb).

## Additional Resources

*   [FiftyOne Enterprise](https://voxel51.com/enterprise)
*   [VoxelGPT](https://github.com/voxel51/voxelgpt)
*   [Plugins](https://voxel51.com/plugins)
*   [Vector Search](https://voxel51.com/blog/the-computer-vision-interface-for-vector-search)
*   [Dataset Zoo](https://docs.voxel51.com/dataset_zoo/index.html)
*   [Model Zoo](https://docs.voxel51.com/model_zoo/index.html)
*   [FiftyOne Brain](https://docs.voxel51.com/brain.html)

## Documentation

Access comprehensive FiftyOne documentation at [fiftyone.ai](https://fiftyone.ai).

*   [Tutorials](https://voxel51.com/docs/fiftyone/tutorials/index.html)
*   [Recipes](https://voxel51.com/docs/fiftyone/recipes/index.html)
*   [Examples](https://github.com/voxel51/fiftyone-examples)
*   [User Guide](https://voxel51.com/docs/fiftyone/user_guide/index.html)
*   [CLI Documentation](https://voxel51.com/docs/fiftyone/cli/index.html)
*   [API Reference](https://voxel51.com/docs/fiftyone/api/fiftyone.html)

## FiftyOne Enterprise

For secure, cloud-based collaboration on large datasets and automated workflows, explore [FiftyOne Enterprise](https://voxel51.com/enterprise).

## FAQ & Troubleshooting

Consult the [common issues page](https://docs.voxel51.com/getting_started/troubleshooting.html) and the [FAQ page](https://docs.voxel51.com/faq/index.html) for troubleshooting. For further assistance, open an issue on [GitHub](https://github.com/voxel51/fiftyone/issues) or contact us on [Discord](https://discord.gg/fiftyone-community).

## Join Our Community

Connect with us on:

[![Discord](https://img.shields.io/badge/Discord-7289DA?logo=discord&logoColor=white)](https://discord.gg/fiftyone-community)
[![Medium](https://img.shields.io/badge/Medium-12100E?logo=medium&logoColor=white)](https://medium.com/voxel51)
[![Twitter](https://img.shields.io/badge/Twitter-1DA1F2?logo=twitter&logoColor=white)](https://twitter.com/voxel51)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?logo=linkedin&logoColor=white)](https://www.linkedin.com/company/voxel51)
[![Facebook](https://img.shields.io/badge/Facebook-1877F2?logo=facebook&logoColor=white)](https://www.facebook.com/voxel51)

**Share your FiftyOne experiences on social media and tag us with @Voxel51 and #FiftyOne!**

## Contributors

FiftyOne is open source and welcomes community contributions! See the [contribution guide](https://github.com/voxel51/fiftyone/blob/develop/CONTRIBUTING.md) to get involved.

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