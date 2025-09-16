![OpenFold Banner](imgs/of_banner.png)
_Figure: Comparison of OpenFold and AlphaFold2 predictions to the experimental structure of PDB 7KDX, chain B._

# OpenFold: Replicating AlphaFold 2 for Protein Structure Prediction

**OpenFold offers a powerful, trainable PyTorch implementation of DeepMind's groundbreaking AlphaFold 2, enabling advancements in protein structure prediction.** 

[View the original repository on GitHub](https://github.com/aqlaboratory/openfold)

## Key Features

*   **Faithful Reproduction:** Built as a direct implementation of AlphaFold 2, ensuring accurate and reliable protein structure predictions.
*   **Trainable Model:**  Offers the ability to train and customize the model, opening avenues for research and optimization.
*   **PyTorch Implementation:** Leverages the flexibility and power of PyTorch for efficient model development and deployment.
*   **Community-Driven:** Open to contributions and improvements from the community, fostering collaborative advancements.

## Documentation and Resources

For detailed information on installation, model inference, and training, please refer to our comprehensive documentation:

*   [OpenFold Documentation](https://openfold.readthedocs.io/en/latest/)

## Copyright and Licensing

*   OpenFold is licensed under the Apache License, Version 2.0.
*   Pretrained parameters (downloaded by the installation script) are licensed under CC BY 4.0.

## Contributing

We welcome contributions! If you encounter any issues or have suggestions, please:

*   [Create an issue](https://github.com/aqlaboratory/openfold/issues) to report problems.
*   Submit pull requests to help improve OpenFold.

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

**Please also cite [AlphaFold](https://www.nature.com/articles/s41586-021-03819-2) and [AlphaFold-Multimer](https://www.biorxiv.org/content/10.1101/2021.10.04.463034v1) if applicable.**