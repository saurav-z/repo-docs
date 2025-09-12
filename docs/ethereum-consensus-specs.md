# Ethereum Proof-of-Stake (PoS) Consensus Specifications

**Dive deep into the technical foundation of Ethereum's Proof-of-Stake, the engine driving secure and sustainable blockchain operations.** Explore the specifications, designed to ensure the robust and scalable functionality of Ethereum. You can find the original repository [here](https://github.com/ethereum/consensus-specs).

[![Join the chat at https://discord.gg/qGpsxSA](https://img.shields.io/badge/chat-on%20discord-blue.svg)](https://discord.gg/qGpsxSA)
[![testgen](https://github.com/ethereum/consensus-specs/actions/workflows/generate_vectors.yml/badge.svg?branch=dev&event=schedule)](https://github.com/ethereum/consensus-specs/actions/workflows/generate_vectors.yml)

## Key Features of the Ethereum PoS Specifications

*   **Comprehensive Specifications:** Detailed documentation covering all aspects of the Ethereum Proof-of-Stake consensus mechanism.
*   **Fork-Specific Definitions:** Specifications are divided by code names, allowing for easier review and understanding of sequential upgrades.
*   **Modular Design:** Specifications are organized into features, allowing for flexible development and consolidation into upgrades.
*   **Robust Testing:** Extensive test suites ensure the correctness and reliability of the specifications, as well as facilitate compliance across clients.
*   **External Integration:** Links to relevant specifications for Beacon APIs, Engine APIs, Beacon Metrics, and Builder Specs.
*   **Ongoing Development:** Specifications for in-development features are also included, allowing for early engagement in the future of the Ethereum blockchain.

## Specifications Overview

This repository contains the core specifications for Ethereum's Proof-of-Stake consensus, organized for clarity and ease of use.

### Stable Specifications

These are the finalized and implemented specifications.

| Seq. | Code Name     | Fork Epoch | Links                                                                        |
| ---- | ------------- | ---------- | ---------------------------------------------------------------------------- |
| 0    | **Phase0**    | `0`        | [Specs](specs/phase0), [Tests](tests/core/pyspec/eth2spec/test/phase0)       |
| 1    | **Altair**    | `74240`    | [Specs](specs/altair), [Tests](tests/core/pyspec/eth2spec/test/altair)       |
| 2    | **Bellatrix** | `144896`   | [Specs](specs/bellatrix), [Tests](tests/core/pyspec/eth2spec/test/bellatrix) |
| 3    | **Capella**   | `194048`   | [Specs](specs/capella), [Tests](tests/core/pyspec/eth2spec/test/capella)     |
| 4    | **Deneb**     | `269568`   | [Specs](specs/deneb), [Tests](tests/core/pyspec/eth2spec/test/deneb)         |
| 5    | **Electra**   | `364032`   | [Specs](specs/electra), [Tests](tests/core/pyspec/eth2spec/test/electra)     |

### In-development Specifications

These are the specifications currently under development.

| Seq. | Code Name | Fork Epoch | Links                                                                |
| ---- | --------- | ---------- | -------------------------------------------------------------------- |
| 6    | **Fulu**  | TBD        | [Specs](specs/fulu), [Tests](tests/core/pyspec/eth2spec/test/fulu)   |
| 7    | **Gloas** | TBD        | [Specs](specs/gloas), [Tests](tests/core/pyspec/eth2spec/test/gloas) |

### Accompanying Documents

*   [SimpleSerialize (SSZ) spec](ssz/simple-serialize.md)
*   [Merkle proof formats](ssz/merkle-proofs.md)
*   [General test format](tests/formats/README.md)

### External Specifications

*   [Beacon APIs](https://github.com/ethereum/beacon-apis)
*   [Engine APIs](https://github.com/ethereum/execution-apis/tree/main/src/engine)
*   [Beacon Metrics](https://github.com/ethereum/beacon-metrics)
*   [Builder Specs](https://github.com/ethereum/builder-specs)

### Reference Tests

*   [Ethereum Proof-of-Stake Consensus Spec Tests](https://github.com/ethereum/consensus-spec-tests)
*   [Nightly Reference Tests](https://github.com/ethereum/consensus-specs/actions/workflows/generate_vectors.yml)
*   Compressed tarballs available for each release [here](https://github.com/ethereum/consensus-spec-tests/releases).

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

*   Prioritize simplicity and minimize complexity.
*   Ensure network liveliness, even during major partitions or node outages.
*   Utilize quantum-resistant components or easily replaceable alternatives.
*   Enable broad validator participation.
*   Minimize hardware requirements for accessibility.

## Useful Resources

*   [Design Rationale](https://notes.ethereum.org/s/rkhCgQteN#)
*   [Phase0 Onboarding Document](https://notes.ethereum.org/s/Bkn3zpwxB)
*   [Combining GHOST and Casper paper](https://arxiv.org/abs/2003.03052)
*   [Specifications viewer (mkdocs)](https://ethereum.github.io/consensus-specs/)
*   [Specifications viewer (jtraglia)](https://jtraglia.github.io/eth-spec-viewer/)
*   [The Eth2 Book](https://eth2book.info)
*   [PySpec Tests](tests/core/pyspec/README.md)