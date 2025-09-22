<!-- Banner Image -->
![OpenFold Banner](imgs/of_banner.png)
_Figure: Comparison of OpenFold and AlphaFold2 predictions to the experimental structure of PDB 7KDX, chain B._

# OpenFold: Replicating AlphaFold 2 for Protein Structure Prediction

OpenFold is a powerful, trainable, and open-source implementation of DeepMind's groundbreaking [AlphaFold 2](https://github.com/deepmind/alphafold), empowering researchers to explore protein structure prediction with unprecedented accuracy.

## Key Features of OpenFold

*   **Faithful Reproduction:** OpenFold accurately replicates the architecture and functionality of AlphaFold 2, providing a solid foundation for research and development.
*   **Trainable Model:**  Fine-tune and adapt OpenFold to your specific needs by training it on custom datasets, opening up possibilities for specialized applications.
*   **Open Source:** Leverage the flexibility of open-source code to modify, extend, and integrate OpenFold into your existing workflows.
*   **Detailed Documentation:** Access comprehensive documentation at [openfold.readthedocs.io](https://openfold.readthedocs.io/en/latest/) for installation, model inference, and training.

## Getting Started

Consult the comprehensive documentation at [openfold.readthedocs.io](https://openfold.readthedocs.io/en/latest/) for detailed instructions on installation, model inference, and training.

## Licensing & Copyright

OpenFold's source code is licensed under the permissive Apache License, Version 2.0.  Pretrained parameters are licensed under CC BY 4.0.

## Contributing

We welcome contributions! If you encounter any issues or have suggestions for improvement, please create an issue or submit a pull request.

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

Remember to also cite [AlphaFold](https://www.nature.com/articles/s41586-021-03819-2) and [AlphaFold-Multimer](https://www.biorxiv.org/content/10.1101/2021.10.04.463034v1) if applicable.

## Further Resources

*   **Documentation:** [openfold.readthedocs.io](https://openfold.readthedocs.io/en/latest/)
*   **Original Repository:** [https://github.com/aqlaboratory/openfold](https://github.com/aqlaboratory/openfold)