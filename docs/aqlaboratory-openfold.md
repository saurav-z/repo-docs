<div align="center">
  <img src="imgs/of_banner.png" alt="OpenFold Banner">
  <p><em>Figure: Comparison of OpenFold and AlphaFold2 predictions to the experimental structure of PDB 7KDX, chain B.</em></p>
</div>

# OpenFold: Reproducing AlphaFold 2 for Protein Structure Prediction

**OpenFold provides a trainable, open-source implementation of DeepMind's groundbreaking AlphaFold 2, empowering researchers with the tools to explore and advance protein structure prediction.** This allows researchers to study protein structures.

[View the original repository on GitHub](https://github.com/aqlaboratory/openfold)

## Key Features

*   **Faithful Reproduction:** OpenFold is built as a faithful PyTorch reproduction of AlphaFold 2, offering a transparent and accessible implementation.
*   **Trainable Model:** Unlike the original, OpenFold allows for training, enabling researchers to fine-tune and adapt the model for specific research needs.
*   **Open Source:** Benefit from a community-driven project, fostering collaboration and innovation in the field of protein structure prediction.
*   **Extensive Documentation:** Detailed instructions for installation, model inference, and training can be found at [openfold.readthedocs.io](https://openfold.readthedocs.io/en/latest/).

## Documentation and Resources

For comprehensive documentation, including installation instructions and guides on model inference and training, please visit our dedicated documentation site: [openfold.readthedocs.io](https://openfold.readthedocs.io/en/latest/).

## Copyright and Licensing

OpenFold utilizes the Apache License, Version 2.0. Pretrained parameters from DeepMind are licensed under the CC BY 4.0 license, which is downloaded during installation.

## Contributing

We welcome contributions! If you encounter any issues or have suggestions, please feel free to create an issue. Pull requests are also encouraged.

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

**Additionally, please cite [AlphaFold](https://www.nature.com/articles/s41586-021-03819-2) and [AlphaFold-Multimer](https://www.biorxiv.org/content/10.1101/2021.10.04.463034v1) when applicable.**