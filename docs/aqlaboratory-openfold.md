[![OpenFold Banner](imgs/of_banner.png)](https://github.com/aqlaboratory/openfold)
_Figure: Comparison of OpenFold and AlphaFold2 predictions to the experimental structure of PDB 7KDX, chain B._

# OpenFold: Reproducing AlphaFold 2 for Protein Structure Prediction

**OpenFold is a PyTorch implementation designed to faithfully replicate DeepMind's AlphaFold 2, providing a trainable platform for protein structure prediction.**

[View the original repository on GitHub](https://github.com/aqlaboratory/openfold)

## Key Features

*   **Faithful Reproduction:** OpenFold aims to replicate the architecture and functionality of AlphaFold 2.
*   **Trainable Model:** Allows users to train and fine-tune the model on custom datasets.
*   **Open Source:** Built upon the Apache License, Version 2.0.
*   **Comprehensive Documentation:** Detailed instructions for installation, inference, and training can be found at [openfold.readthedocs.io](https://openfold.readthedocs.io/en/latest/).

## Getting Started

Explore the documentation to get started with OpenFold:

*   **Installation:** Follow the instructions on the documentation site to install OpenFold.
*   **Model Inference:** Learn how to predict protein structures using the pre-trained model.
*   **Training:** Discover how to train the model on your own data.

## Copyright and Licensing

OpenFold's source code is licensed under the Apache License, Version 2.0. Pretrained parameters from AlphaFold are licensed under the CC BY 4.0 license, which is downloaded during installation.

## Contributing

We encourage community contributions! If you find any issues or have suggestions, feel free to:

*   **Create an issue:** Report bugs or suggest features.
*   **Submit pull requests:** Contribute code and improvements.

## Citing OpenFold

If you use OpenFold in your research, please cite the following paper:

```bibtex
@article {Ahdritz2022.11.20.517210,
	author = {Ahdritz, Gustaf and Bouatta, Nazim and Floristean, Christina and Kadyan, Sachin and Xia, Qinghui and Gerecke, William and O{\textquoteright}Donnell, Timothy J and Berenberg, Daniel and Fisk, Ian and Zanichelli, Niccolò and Zhang, Bo and Nowaczynski, Arkadiusz and Wang, Bei and Stepniewska-Dziubinska, Marta M and Zhang, Shang and Ojewole, Adegoke and Guney, Murat Efe and Biderman, Stella and Watkins, Andrew M and Ra, Stephen and Lorenzo, Pablo Ribalta and Nivon, Lucas and Weitzner, Brian and Ban, Yih-En Andrew and Sorger, Peter K and Mostaque, Emad and Zhang, Zhao and Bonneau, Richard and AlQuraishi, Mohammed},
	title = {{O}pen{F}old: {R}etraining {A}lpha{F}old2 yields new insights into its learning mechanisms and capacity for generalization},
	elocation-id = {2022.11.20.517210},
	year = {2022},
	doi = {10.1101/2022.11.20.517210},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/10.1101/2022.11.20.517210},
	eprint = {https://www.biorxiv.org/content/early/2022/11/22/2022.11.20.517210.full.pdf},
	journal = {bioRxiv}
}
```

If you utilize OpenProteinSet, kindly cite:

```bibtex
@misc{ahdritz2023openproteinset,
      title={{O}pen{P}rotein{S}et: {T}raining data for structural biology at scale}, 
      author={Gustaf Ahdritz and Nazim Bouatta and Sachin Kadyan and Lukas Jarosch and Daniel Berenberg and Ian Fisk and Andrew M. Watkins and Stephen Ra and Richard Bonneau and Mohammed AlQuraishi},
      year={2023},
      eprint={2308.05326},
      archivePrefix={arXiv},
      primaryClass={q-bio.BM}
}
```

Any work that cites OpenFold should also cite [AlphaFold](https://www.nature.com/articles/s41586-021-03819-2) and [AlphaFold-Multimer](https://www.biorxiv.org/content/10.1101/2021.10.04.463034v1) if applicable.