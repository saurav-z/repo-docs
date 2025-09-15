# Ethereum Proof-of-Stake Consensus Specifications: Powering the Future of Ethereum

Explore the definitive specifications that govern Ethereum's [proof-of-stake](https://ethereum.org/en/developers/docs/consensus-mechanisms/pos/) consensus mechanism and contribute to the evolution of the world's leading decentralized blockchain.

[![Join the chat at https://discord.gg/qGpsxSA](https://img.shields.io/badge/chat-on%20discord-blue.svg)](https://discord.gg/qGpsxSA)
[![testgen](https://github.com/ethereum/consensus-specs/actions/workflows/generate_vectors.yml/badge.svg?branch=dev&event=schedule)](https://github.com/ethereum/consensus-specs/actions/workflows/generate_vectors.yml)

This repository serves as the central hub for the development and documentation of the Ethereum proof-of-stake (PoS) specifications, providing a comprehensive resource for developers, researchers, and anyone interested in the technical details of Ethereum's consensus layer. Discussions, proposals, and specifications are managed through issues and pull requests, ensuring a collaborative and transparent development process.

## Key Features

*   **Comprehensive Specifications:** Detailed documentation of all aspects of the Ethereum PoS consensus mechanism.
*   **Versioned Specifications:** Clear delineation of stable and in-development specifications, enabling clarity on the status of each iteration.
*   **Test Vectors:** Integration of test suites to ensure the reliability and interoperability of client implementations.
*   **Modular Design:** Specifications are divided into features, allowing for parallel development and sequential upgrades.
*   **Collaborative Development:**  Engage with the community through issues and pull requests to shape the future of Ethereum.

## Specifications

The core specifications are organized by "forks" or upgrades, with each one addressing particular improvements.

### Stable Specifications

| Seq. | Code Name     | Fork Epoch | Links                                                                        |
| ---- | ------------- | ---------- | ---------------------------------------------------------------------------- |
| 0    | **Phase0**    | `0`        | [Specs](specs/phase0), [Tests](tests/core/pyspec/eth2spec/test/phase0)       |
| 1    | **Altair**    | `74240`    | [Specs](specs/altair), [Tests](tests/core/pyspec/eth2spec/test/altair)       |
| 2    | **Bellatrix** | `144896`   | [Specs](specs/bellatrix), [Tests](tests/core/pyspec/eth2spec/test/bellatrix) |
| 3    | **Capella**   | `194048`   | [Specs](specs/capella), [Tests](tests/core/pyspec/eth2spec/test/capella)     |
| 4    | **Deneb**     | `269568`   | [Specs](specs/deneb), [Tests](tests/core/pyspec/eth2spec/test/deneb)         |
| 5    | **Electra**   | `364032`   | [Specs](specs/electra), [Tests](tests/core/pyspec/eth2spec/test/electra)     |

### In-development Specifications

| Seq. | Code Name | Fork Epoch | Links                                                                |
| ---- | --------- | ---------- | -------------------------------------------------------------------- |
| 6    | **Fulu**  | TBD        | [Specs](specs/fulu), [Tests](tests/core/pyspec/eth2spec/test/fulu)   |
| 7    | **Gloas** | TBD        | [Specs](specs/gloas), [Tests](tests/core/pyspec/eth2spec/test/gloas) |

### Accompanying Documents

*   [SimpleSerialize (SSZ) spec](ssz/simple-serialize.md)
*   [Merkle proof formats](ssz/merkle-proofs.md)
*   [General test format](tests/formats/README.md)

### External Specifications

Additional specifications and standards outside of requisite client
functionality can be found in the following repositories:

*   [Beacon APIs](https://github.com/ethereum/beacon-apis)
*   [Engine APIs](https://github.com/ethereum/execution-apis/tree/main/src/engine)
*   [Beacon Metrics](https://github.com/ethereum/beacon-metrics)
*   [Builder Specs](https://github.com/ethereum/builder-specs)

### Reference Tests

Reference tests built from the executable Python spec are available in the
[Ethereum Proof-of-Stake Consensus Spec Tests](https://github.com/ethereum/consensus-spec-tests)
repository. Compressed tarballs are available for each release
[here](https://github.com/ethereum/consensus-spec-tests/releases). Nightly
reference tests are available
[here](https://github.com/ethereum/consensus-specs/actions/workflows/generate_vectors.yml).

## Getting Started

### Installation and Usage

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/ethereum/consensus-specs.git
    ```

2.  **Navigate to the directory:**

    ```bash
    cd consensus-specs
    ```

3.  **View the help output:**

    ```bash
    make help
    ```

## Design Goals

The Ethereum PoS specifications are designed with the following key goals in mind:

*   **Simplicity:** Prioritizing clarity and ease of understanding, even at the expense of marginal efficiency gains.
*   **Robustness:** Ensuring network stability and liveness, even during significant partitions or validator outages.
*   **Quantum Resistance:** Incorporating technologies that are either inherently quantum-resistant or easily replaceable with quantum-secure alternatives.
*   **Decentralization:** Allowing for a large number of validators to participate.
*   **Accessibility:** Minimizing hardware requirements to enable participation on standard consumer devices.

## Useful Resources

*   [Design Rationale](https://notes.ethereum.org/s/rkhCgQteN#)
*   [Phase0 Onboarding Document](https://notes.ethereum.org/s/Bkn3zpwxB)
*   [Combining GHOST and Casper paper](https://arxiv.org/abs/2003.03052)
*   [Specifications viewer (mkdocs)](https://ethereum.github.io/consensus-specs/)
*   [Specifications viewer (jtraglia)](https://jtraglia.github.io/eth-spec-viewer/)
*   [The Eth2 Book](https://eth2book.info)
*   [PySpec Tests](tests/core/pyspec/README.md)

For more information and to contribute, visit the [original repository](https://github.com/ethereum/consensus-specs).