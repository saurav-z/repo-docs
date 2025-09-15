![OpenFold Banner](imgs/of_banner.png)
_Figure: Comparison of OpenFold and AlphaFold2 predictions to the experimental structure of PDB 7KDX, chain B._

# OpenFold: Replicating AlphaFold 2 for Protein Structure Prediction

**OpenFold is a PyTorch-based, trainable reproduction of DeepMind's groundbreaking AlphaFold 2, empowering researchers to explore and advance protein structure prediction.**

[![Documentation](https://img.shields.io/badge/documentation-openfold.readthedocs.io-blue)](https://openfold.readthedocs.io/en/latest/)
[![GitHub](https://img.shields.io/github/stars/aqlaboratory/openfold?style=social)](https://github.com/aqlaboratory/openfold)

## Key Features

*   **Faithful Reproduction:** OpenFold is built to mirror the architecture and functionality of AlphaFold 2.
*   **Trainable Model:** Enables users to fine-tune the model with custom datasets and explore its learning mechanisms.
*   **Open Source:** Benefit from the Apache 2.0 license, allowing for wide use and modification.
*   **Comprehensive Documentation:** Detailed guides for installation, inference, and training are available.

## Documentation & Getting Started

For detailed instructions on installing and using OpenFold, including model inference and training procedures, please refer to our comprehensive documentation: [openfold.readthedocs.io](https://openfold.readthedocs.io/en/latest/)

## Copyright & Licensing

OpenFold's source code is licensed under the permissive [Apache Licence, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0). Note that the pretrained parameters are licensed under the CC BY 4.0 license.

## Contributing

We encourage community contributions! Feel free to submit issues for any problems encountered or pull requests for improvements.

## Citing OpenFold

If you utilize OpenFold in your research, please cite our paper:

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
If you use OpenProteinSet, please also cite:

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

When citing OpenFold, please also cite [AlphaFold](https://www.nature.com/articles/s41586-021-03819-2) and [AlphaFold-Multimer](https://www.biorxiv.org/content/10.1101/2021.10.04.463034v1) if applicable.

## Get Involved

Explore the source code, contribute, and stay up-to-date on the latest developments by visiting the [OpenFold repository](https://github.com/aqlaboratory/openfold).