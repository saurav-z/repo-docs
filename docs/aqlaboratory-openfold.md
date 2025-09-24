![OpenFold Banner](imgs/of_banner.png)
_Figure: Comparison of OpenFold and AlphaFold2 predictions to the experimental structure of PDB 7KDX, chain B._

# OpenFold: Replicating AlphaFold 2 for Protein Structure Prediction

**OpenFold is a trainable and open-source PyTorch implementation of DeepMind's groundbreaking AlphaFold 2, empowering researchers with accessible protein structure prediction capabilities.**

## Key Features

*   **Faithful Reproduction:** OpenFold provides a close replication of the AlphaFold 2 architecture.
*   **Trainable Model:** The PyTorch implementation allows for training and fine-tuning the model on custom datasets.
*   **Open Source:** Explore and contribute to the code, fostering collaboration and innovation in protein structure prediction.

## Getting Started

For comprehensive instructions on installation, model inference, and training, please refer to our detailed documentation: [openfold.readthedocs.io](https://openfold.readthedocs.io/en/latest/).

## Important Information

*   **Copyright & Licensing:** OpenFold's source code is licensed under the Apache License, Version 2.0.  The pretrained parameters, obtained during installation, are licensed under CC BY 4.0.
*   **Original README:**  You can find much of the original README content [here](https://github.com/aqlaboratory/openfold/blob/main/docs/source/original_readme.md).

## Contributing

We welcome contributions from the community! Feel free to submit issues for bug reports and feature requests, and submit pull requests to improve OpenFold.

## Citing OpenFold

If you use OpenFold in your research, please cite our paper:

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

If you use OpenProteinSet, also cite:

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

**Remember to also cite AlphaFold ([https://www.nature.com/articles/s41586-021-03819-2](https://www.nature.com/articles/s41586-021-03819-2)) and AlphaFold-Multimer ([https://www.biorxiv.org/content/10.1101/2021.10.04.463034v1](https://www.biorxiv.org/content/10.1101/2021.10.04.463034v1)) if applicable.**

##  [Explore the OpenFold Repository on GitHub](https://github.com/aqlaboratory/openfold)