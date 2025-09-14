# Ethereum Proof-of-Stake Consensus Specifications: The Definitive Guide

**Dive deep into the specifications that power Ethereum's groundbreaking transition to Proof-of-Stake.** For the latest updates and detailed specifications, visit the [original repository](https://github.com/ethereum/consensus-specs).

## Key Features

*   **Comprehensive Specifications:** Access detailed specifications for Ethereum's Proof-of-Stake consensus mechanism, including core functionalities and future upgrades.
*   **Stable and In-Development Specifications:** Explore both the currently implemented and upcoming features of the Ethereum consensus layer, allowing developers to stay ahead of the curve.
*   **Versioned Releases:** Follow along with the sequential upgrades that have been implemented to improve the overall network.
*   **Accompanying Documents:** Find comprehensive documentation for SimpleSerialize (SSZ), Merkle proofs, and general test formats.
*   **External Specifications:** Explore specifications and standards that are external but related to the core functionality (beacon APIs, engine APIs, etc).

## Specifications Overview

The Ethereum Proof-of-Stake specifications are organized into sequential upgrades that are researched, developed, and tested in parallel before being integrated into the network.

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

*   [Beacon APIs](https://github.com/ethereum/beacon-apis)
*   [Engine APIs](https://github.com/ethereum/execution-apis/tree/main/src/engine)
*   [Beacon Metrics](https://github.com/ethereum/beacon-metrics)
*   [Builder Specs](https://github.com/ethereum/builder-specs)

### Reference Tests

*   [Ethereum Proof-of-Stake Consensus Spec Tests](https://github.com/ethereum/consensus-spec-tests)
*   Nightly tests: [here](https://github.com/ethereum/consensus-specs/actions/workflows/generate_vectors.yml)

## Getting Started

### Installation

Clone the repository:

```bash
git clone https://github.com/ethereum/consensus-specs.git
```

Navigate to the directory:

```bash
cd consensus-specs
```

### Usage

View available commands:

```bash
make help
```

## Design Goals

The Ethereum Proof-of-Stake specifications are built with the following key design goals:

*   **Simplicity:** Minimize complexity for easier understanding and maintenance.
*   **Resilience:** Ensure network stability during partitions and node failures.
*   **Quantum Security:** Prioritize quantum-resistant components or easy swappability.
*   **Scalability:** Facilitate a high validator participation rate.
*   **Accessibility:** Minimize hardware requirements for broad participation.

## Useful Resources

*   [Design Rationale](https://notes.ethereum.org/s/rkhCgQteN#)
*   [Phase0 Onboarding Document](https://notes.ethereum.org/s/Bkn3zpwxB)
*   [Combining GHOST and Casper paper](https://arxiv.org/abs/2003.03052)
*   [Specifications viewer (mkdocs)](https://ethereum.github.io/consensus-specs/)
*   [Specifications viewer (jtraglia)](https://jtraglia.github.io/eth-spec-viewer/)
*   [The Eth2 Book](https://eth2book.info)
*   [PySpec Tests](tests/core/pyspec/README.md)