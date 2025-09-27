[![OpenFold Banner](imgs/of_banner.png)](https://github.com/aqlaboratory/openfold)

_Figure: Comparison of OpenFold and AlphaFold2 predictions to the experimental structure of PDB 7KDX, chain B._

# OpenFold: Replicating AlphaFold 2 for Protein Structure Prediction

**OpenFold** is a powerful, trainable PyTorch implementation of DeepMind's groundbreaking [AlphaFold 2](https://github.com/deepmind/alphafold) that empowers researchers to explore protein structure prediction and its underlying mechanisms.

## Key Features

*   **Faithful Reproduction:** Built to mirror the architecture and functionality of AlphaFold 2.
*   **Trainable Models:**  Enables researchers to fine-tune and experiment with the model.
*   **Open Source:** Freely available under the Apache License, Version 2.0, promoting collaboration and innovation.
*   **Detailed Documentation:** Comprehensive documentation available at [openfold.readthedocs.io](https://openfold.readthedocs.io/en/latest/), including installation and usage instructions.

## Getting Started

For detailed instructions on installation, model inference, and training, please refer to our comprehensive documentation: [openfold.readthedocs.io](https://openfold.readthedocs.io/en/latest/).

## Licensing and Copyright

OpenFold's source code is licensed under the Apache License, Version 2.0.  Pretrained parameters, available in `openfold/resources/params`, are licensed under the CC BY 4.0 license.

## Contributing

We encourage community involvement!  Feel free to submit issues or pull requests to help improve OpenFold.

## Citation

If you use OpenFold in your research, please cite the following papers:

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

If you use OpenProteinSet, please cite:

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

**Also, please cite [AlphaFold](https://www.nature.com/articles/s41586-021-03819-2) and [AlphaFold-Multimer](https://www.biorxiv.org/content/10.1101/2021.10.04.463034v1) if applicable.**

## Original Repository

Explore the full source code and contribute at the [OpenFold GitHub repository](https://github.com/aqlaboratory/openfold).