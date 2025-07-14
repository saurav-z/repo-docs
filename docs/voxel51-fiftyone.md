<div align="center">
<p align="center">
<!-- prettier-ignore -->
<img src="https://user-images.githubusercontent.com/25985824/106288517-2422e000-6216-11eb-871d-26ad2e7b1e59.png" height="55px"> &nbsp;
<img src="https://user-images.githubusercontent.com/25985824/106288518-24bb7680-6216-11eb-8f10-60052c519586.png" height="50px">
</p>
</div>

# FiftyOne: The Open-Source Toolkit for Computer Vision Data Management

FiftyOne empowers you to build high-quality datasets and computer vision models with its powerful visualization, analysis, and data curation capabilities.  Check out the [original repository](https://github.com/voxel51/fiftyone) to learn more.

---

**Key Features:**

*   **Dataset Visualization:** Visualize images, videos, and associated labels in an intuitive visual interface.
*   **Embeddings Exploration:** Discover insights with easy-to-use embedding exploration.
*   **Model Analysis and Improvement:** Evaluate model performance, identify failures, and fine-tune models effectively.
*   **Advanced Data Curation:** Quickly identify and resolve data issues, annotation errors, and edge cases.
*   **Rich Integrations:** Seamlessly integrates with popular deep learning libraries like PyTorch, Hugging Face, and Ultralytics.
*   **Extensible and Customizable:** Tailor FiftyOne to your specific needs with its open and extensible design.

---

**Quickstart**

Get started by running the following Python code:

```python
import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset("quickstart")
session = fo.launch_app(dataset)
```

Explore the [quickstart Colab notebook](https://colab.research.google.com/github/voxel51/fiftyone-examples/blob/master/examples/quickstart.ipynb) for common workflows.

---

**Installation**

```bash
pip install fiftyone
```

See [installation details](https://voxel51.com/docs/fiftyone/getting_started/install.html) for more information.

---

**Resources**

*   [Website](https://voxel51.com/fiftyone)
*   [Docs](https://voxel51.com/docs/fiftyone)
*   [Tutorials](https://voxel51.com/docs/fiftyone/tutorials/index.html)
*   [Examples](https://github.com/voxel51/fiftyone-examples)
*   [Blog](https://voxel51.com/blog/)
*   [Community](https://discord.gg/fiftyone-community)
*   [FiftyOne Enterprise](http://voxel51.com/enterprise)

---

**Join the Community**

*   [Discord](https://discord.gg/fiftyone-community)
*   [Medium](https://medium.com/voxel51)
*   [Twitter](https://twitter.com/voxel51)
*   [LinkedIn](https://www.linkedin.com/company/voxel51)
*   [Facebook](https://www.facebook.com/voxel51)

**Tag us with @Voxel51 and #FiftyOne to share your projects!**

---

**Contributors**

FiftyOne is an open-source project, and contributions are welcome!
<a href="https://github.com/voxel51/fiftyone/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=voxel51/fiftyone" />
</a>

---

**Citation**

```bibtex
@article{moore2020fiftyone,
  title={FiftyOne},
  author={Moore, B. E. and Corso, J. J.},
  journal={GitHub. Note: https://github.com/voxel51/fiftyone},
  year={2020}
}