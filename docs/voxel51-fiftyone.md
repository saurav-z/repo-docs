<div align="center">
<p align="center">

<!-- prettier-ignore -->
<img src="https://user-images.githubusercontent.com/25985824/106288517-2422e000-6216-11eb-871d-26ad2e7b1e59.png" height="55px"> &nbsp;
<img src="https://user-images.githubusercontent.com/25985824/106288518-24bb7680-6216-11eb-8f10-60052c519586.png" height="50px">

</p>
</div>

# FiftyOne: Build High-Quality Datasets & Computer Vision Models

**FiftyOne** is the open-source tool that empowers you to build superior datasets and computer vision models. ([View on GitHub](https://github.com/voxel51/fiftyone))

---

**Key Features:**

*   **[Visualize Complex Datasets](https://docs.voxel51.com/user_guide/app.html):** Easily explore images, videos, and associated labels in a powerful visual interface.
*   **[Explore Embeddings:](https://docs.voxel51.com/user_guide/app.html#embeddings-panel)** Select points of interest and view the corresponding samples/labels.
*   **[Analyze and Improve Models:](https://docs.voxel51.com/user_guide/evaluation.html)** Evaluate model performance, identify failure modes, and fine-tune your models.
*   **[Advanced Data Curation:](https://docs.voxel51.com/brain.html)** Quickly find and fix data issues, annotation errors, and edge cases.
*   **[Rich Integrations:](https://docs.voxel51.com/integrations/index.html)** Works with popular deep learning libraries like PyTorch, Hugging Face, Ultralytics, and more.
*   **[Open and Extensible:](https://docs.voxel51.com/plugins/index.html)** Customize and extend FiftyOne to fit your specific needs.

**Get Started**

Install with pip:

```bash
pip install fiftyone
```

And explore a quickstart dataset:

```python
import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset("quickstart")
session = fo.launch_app(dataset)
```
See more details in the original [README](https://github.com/voxel51/fiftyone).

**Explore FiftyOne Resources:**

*   [Website](https://voxel51.com/fiftyone)
*   [Docs](https://voxel51.com/docs/fiftyone)
*   [Try it Now](https://colab.research.google.com/github/voxel51/fiftyone-examples/blob/master/examples/quickstart.ipynb)
*   [Tutorials](https://voxel51.com/docs/fiftyone/tutorials/index.html)
*   [Examples](https://github.com/voxel51/fiftyone-examples)
*   [Blog](https://voxel51.com/blog/)
*   [Community](https://discord.gg/fiftyone-community)

---

**FiftyOne Enterprise:**

Scale to production-grade, collaborative, cloud-native enterprise workloads with [FiftyOne Enterprise](http://voxel51.com/enterprise).

**Join the Community:**

*   [Discord](https://discord.gg/fiftyone-community)
*   [Medium](https://medium.com/voxel51)
*   [Twitter](https://twitter.com/voxel51)
*   [LinkedIn](https://www.linkedin.com/company/voxel51)
*   [Facebook](https://www.facebook.com/voxel51)

**Contribute:**

FiftyOne is open source and welcomes community contributions.  See the [contribution guide](https://github.com/voxel51/fiftyone/blob/develop/CONTRIBUTING.md) to learn how to get involved.

**Citation:**

```bibtex
@article{moore2020fiftyone,
  title={FiftyOne},
  author={Moore, B. E. and Corso, J. J.},
  journal={GitHub. Note: https://github.com/voxel51/fiftyone},
  year={2020}
}