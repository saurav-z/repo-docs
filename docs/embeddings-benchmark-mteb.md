# Massive Text Embedding Benchmark (MTEB)

**Evaluate and Compare Text Embedding Models Across Diverse Tasks.**  [Explore the MTEB Repository](https://github.com/embeddings-benchmark/mteb)

MTEB provides a comprehensive benchmark for evaluating text embedding models, offering a standardized and reproducible framework for assessing performance across various tasks and datasets.

## Key Features:

*   **Extensive Task Coverage:** Supports a wide range of tasks, including classification, retrieval, clustering, and pair classification, across multiple modalities and languages.
*   **Easy-to-Use Interface:** Simple installation and intuitive API for evaluating models with minimal setup.
*   **Leaderboard:** Public leaderboard to track model performance and compare against state-of-the-art results.
*   **Reproducibility:** Provides clear instructions and tools for reproducing results, facilitating research and development.
*   **Customization:** Allows for the evaluation of custom models and tasks, enabling flexible experimentation.
*   **Detailed Documentation:** Comprehensive documentation, including guides, tutorials, and examples.

## Quickstart

### Installation

```bash
pip install mteb
```

### Example Usage

```python
import mteb
from sentence_transformers import SentenceTransformer

# Define the sentence-transformers model name
model_name = "average_word_embeddings_komninos"

model = mteb.get_model(model_name) # if the model is not implemented in MTEB it will be eq. to SentenceTransformer(model_name)
tasks = mteb.get_tasks(tasks=["Banking77Classification"])
evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, output_folder=f"results/{model_name}")
```

```bash
mteb available_tasks # list _all_ available tasks

mteb run -m sentence-transformers/all-MiniLM-L6-v2 \
    -t Banking77Classification  \
    --verbosity 3

# if nothing is specified default to saving the results in the results/{model_name} folder
```

## Key Resources

*   **[Leaderboard](https://huggingface.co/spaces/mteb/leaderboard):** Interactive platform to view and compare model performance.
*   **[Documentation](docs/usage/usage.md):** Comprehensive documentation covering installation, usage, and customization options.
*   **[Tasks Overview](docs/tasks.md):** Detailed information on the available tasks.
*   **[Benchmarks Overview](docs/benchmarks.md):** Information on available benchmarks.
*   **[Contributing](CONTRIBUTING.md):** Guidelines for contributing to the MTEB project.

## Citing

MTEB was introduced in "[MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316)", and heavily expanded in "[MMTEB: Massive Multilingual Text Embedding Benchmark](https://arxiv.org/abs/2502.13595)". When using `mteb`, we recommend that you cite both articles.

<details>
  <summary> Bibtex Citation (click to unfold) </summary>


```bibtex
@article{muennighoff2022mteb,
  author = {Muennighoff, Niklas and Tazi, Nouamane and Magne, Lo{\"\i}c and Reimers, Nils},
  title = {MTEB: Massive Text Embedding Benchmark},
  publisher = {arXiv},
  journal={arXiv preprint arXiv:2210.07316},
  year = {2022}
  url = {https://arxiv.org/abs/2210.07316},
  doi = {10.48550/ARXIV.2210.07316},
}

@article{enevoldsen2025mmtebmassivemultilingualtext,
  title={MMTEB: Massive Multilingual Text Embedding Benchmark},
  author={Kenneth Enevoldsen and Isaac Chung and Imene Kerboua and Márton Kardos and Ashwin Mathur and David Stap and Jay Gala and Wissam Siblini and Dominik Krzemiński and Genta Indra Winata and Saba Sturua and Saiteja Utpala and Mathieu Ciancone and Marion Schaeffer and Gabriel Sequeira and Diganta Misra and Shreeya Dhakal and Jonathan Rystrøm and Roman Solomatin and Ömer Çağatan and Akash Kundu and Martin Bernstorff and Shitao Xiao and Akshita Sukhlecha and Bhavish Pahwa and Rafał Poświata and Kranthi Kiran GV and Shawon Ashraf and Daniel Auras and Björn Plüster and Jan Philipp Harries and Loïc Magne and Isabelle Mohr and Mariya Hendriksen and Dawei Zhu and Hippolyte Gisserot-Boukhlef and Tom Aarsen and Jan Kostkan and Konrad Wojtasik and Taemin Lee and Marek Šuppa and Crystina Zhang and Roberta Rocca and Mohammed Hamdy and Andrianos Michail and John Yang and Manuel Faysse and Aleksei Vatolin and Nandan Thakur and Manan Dey and Dipam Vasani and Pranjal Chitale and Simone Tedeschi and Nguyen Tai and Artem Snegirev and Michael Günther and Mengzhou Xia and Weijia Shi and Xing Han Lù and Jordan Clive and Gayatri Krishnakumar and Anna Maksimova and Silvan Wehrli and Maria Tikhonova and Henil Panchal and Aleksandr Abramov and Malte Ostendorff and Zheng Liu and Simon Clematide and Lester James Miranda and Alena Fenogenova and Guangyu Song and Ruqiya Bin Safi and Wen-Ding Li and Alessia Borghini and Federico Cassano and Hongjin Su and Jimmy Lin and Howard Yen and Lasse Hansen and Sara Hooker and Chenghao Xiao and Vaibhav Adlakha and Orion Weller and Siva Reddy and Niklas Muennighoff},
  publisher = {arXiv},
  journal={arXiv preprint arXiv:2502.13595},
  year={2025},
  url={https://arxiv.org/abs/2502.13595},
  doi = {10.48550/arXiv.2502.13595},
}
```
</details>


If you use any of the specific benchmarks, we also recommend that you cite the authors.

```py
benchmark = mteb.get_benchmark("MTEB(eng, v2)")
benchmark.citation # get citation for a specific benchmark

# you can also create a table of the task for the appendix using:
benchmark.tasks.to_latex()
```

Some of these amazing publications include (ordered chronologically):
- Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muennighoff. "[C-Pack: Packaged Resources To Advance General Chinese Embedding](https://arxiv.org/abs/2309.07597)" arXiv 2023
- Michael Günther, Jackmin Ong, Isabelle Mohr, Alaeddine Abdessalem, Tanguy Abel, Mohammad Kalim Akram, Susana Guzman, Georgios Mastrapas, Saba Sturua, Bo Wang, Maximilian Werk, Nan Wang, Han Xiao. "[Jina Embeddings 2: 8192-Token General-Purpose Text Embeddings for Long Documents](https://arxiv.org/abs/2310.19923)" arXiv 2023
- Silvan Wehrli, Bert Arnrich, Christopher Irrgang. "[German Text Embedding Clustering Benchmark](https://arxiv.org/abs/2401.02709)" arXiv 2024
- Orion Weller, Benjamin Chang, Sean MacAvaney, Kyle Lo, Arman Cohan, Benjamin Van Durme, Dawn Lawrie, Luca Soldaini. "[FollowIR: Evaluating and Teaching Information Retrieval Models to Follow Instructions](https://arxiv.org/abs/2403.15246)" arXiv 2024
- Dawei Zhu, Liang Wang, Nan Yang, Yifan Song, Wenhao Wu, Furu Wei, Sujian Li. "[LongEmbed: Extending Embedding Models for Long Context Retrieval](https://arxiv.org/abs/2404.12096)" arXiv 2024
- Kenneth Enevoldsen, Márton Kardos, Niklas Muennighoff, Kristoffer Laigaard Nielbo. "[The Scandinavian Embedding Benchmarks: Comprehensive Assessment of Multilingual and Monolingual Text Embedding](https://arxiv.org/abs/2406.02396)" arXiv 2024
- Ali Shiraee Kasmaee, Mohammad Khodadad, Mohammad Arshi Saloot, Nick Sherck, Stephen Dokas, Hamidreza Mahyar, Soheila Samiee. "[ChemTEB: Chemical Text Embedding Benchmark, an Overview of Embedding Models Performance & Efficiency on a Specific Domain](https://arxiv.org/abs/2412.00532)" arXiv 2024
- Chenghao Xiao, Isaac Chung, Imene Kerboua, Jamie Stirling, Xin Zhang, Márton Kardos, Roman Solomatin, Noura Al Moubayed, Kenneth Enevoldsen, Niklas Muennighoff. "[MIEB: Massive Image Embedding Benchmark](https://arxiv.org/abs/2504.10471)" arXiv 2025