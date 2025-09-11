# Ethereum Proof-of-Stake (PoS) Consensus Specifications

**Dive deep into the technical heart of Ethereum 2.0 with the definitive specifications for its Proof-of-Stake consensus mechanism.** ([Original Repository](https://github.com/ethereum/consensus-specs))

[![Join the chat on Discord](https://img.shields.io/badge/chat-on%20discord-blue.svg)](https://discord.gg/qGpsxSA)
[![Test Generation Workflow](https://github.com/ethereum/consensus-specs/actions/workflows/generate_vectors.yml/badge.svg?branch=dev&event=schedule)](https://github.com/ethereum/consensus-specs/actions/workflows/generate_vectors.yml)

This repository serves as the central source for understanding and implementing the Ethereum proof-of-stake specifications.  It provides the core specifications for developers and researchers looking to build and understand Ethereum clients.

## Key Features

*   **Comprehensive Specifications:** Detailed specifications for all phases of the Ethereum PoS consensus mechanism.
*   **Clear Organization:** Specifications are divided into features and consolidated into sequential upgrades.
*   **Regular Updates:**  Specifications are actively maintained and updated through a collaborative process involving issues and pull requests.
*   **Extensive Testing:** Includes a robust suite of tests to ensure the correctness and interoperability of client implementations.
*   **Open Source:**  Freely available under an open-source license, fostering collaboration and community involvement.

## Specifications Overview

### Stable Specifications

The following are the stable specifications that have been implemented in Ethereum:

| Seq. | Code Name     | Fork Epoch | Links                                                                        |
| ---- | ------------- | ---------- | ---------------------------------------------------------------------------- |
| 0    | **Phase0**    | `0`        | [Specs](specs/phase0), [Tests](tests/core/pyspec/eth2spec/test/phase0)       |
| 1    | **Altair**    | `74240`    | [Specs](specs/altair), [Tests](tests/core/pyspec/eth2spec/test/altair)       |
| 2    | **Bellatrix** | `144896`   | [Specs](specs/bellatrix), [Tests](tests/core/pyspec/eth2spec/test/bellatrix) |
| 3    | **Capella**   | `194048`   | [Specs](specs/capella), [Tests](tests/core/pyspec/eth2spec/test/capella)     |
| 4    | **Deneb**     | `269568`   | [Specs](specs/deneb), [Tests](tests/core/pyspec/eth2spec/test/deneb)         |
| 5    | **Electra**   | `364032`   | [Specs](specs/electra), [Tests](tests/core/pyspec/eth2spec/test/electra)     |

### In-development Specifications

Ongoing development is focused on future upgrades to enhance the Ethereum network:

| Seq. | Code Name | Fork Epoch | Links                                                                |
| ---- | --------- | ---------- | -------------------------------------------------------------------- |
| 6    | **Fulu**  | TBD        | [Specs](specs/fulu), [Tests](tests/core/pyspec/eth2spec/test/fulu)   |
| 7    | **Gloas** | TBD        | [Specs](specs/gloas), [Tests](tests/core/pyspec/eth2spec/test/gloas) |

### Accompanying Documents

These documents provide more in-depth insights into specific components:

*   [SimpleSerialize (SSZ) spec](ssz/simple-serialize.md)
*   [Merkle proof formats](ssz/merkle-proofs.md)
*   [General test format](tests/formats/README.md)

### External Specifications

Additional related specifications and standards can be found in these repositories:

*   [Beacon APIs](https://github.com/ethereum/beacon-apis)
*   [Engine APIs](https://github.com/ethereum/execution-apis/tree/main/src/engine)
*   [Beacon Metrics](https://github.com/ethereum/beacon-metrics)
*   [Builder Specs](https://github.com/ethereum/builder-specs)

### Reference Tests

*   [Ethereum Proof-of-Stake Consensus Spec Tests](https://github.com/ethereum/consensus-spec-tests): Provides comprehensive tests.
*   Nightly Reference Tests: Available [here](https://github.com/ethereum/consensus-specs/actions/workflows/generate_vectors.yml).
*   Compressed tarballs for each release can be found [here](https://github.com/ethereum/consensus-spec-tests/releases).

## Getting Started

### Installation

Clone the repository using:

```bash
git clone https://github.com/ethereum/consensus-specs.git
```

### Usage

Navigate into the repository directory:

```bash
cd consensus-specs
```

View available commands:

```bash
make help
```

## Design Goals

The Ethereum PoS consensus specifications are designed to achieve the following:

*   Minimize Complexity
*   Network Resilience
*   Quantum Security
*   Validator Participation
*   Low Hardware Requirements

## Resources

*   [Design Rationale](https://notes.ethereum.org/s/rkhCgQteN#)
*   [Phase0 Onboarding Document](https://notes.ethereum.org/s/Bkn3zpwxB)
*   [Combining GHOST and Casper paper](https://arxiv.org/abs/2003.03052)
*   [Specifications viewer (mkdocs)](https://ethereum.github.io/consensus-specs/)
*   [Specifications viewer (jtraglia)](https://jtraglia.github.io/eth-spec-viewer/)
*   [The Eth2 Book](https://eth2book.info)
*   [PySpec Tests](tests/core/pyspec/README.md)