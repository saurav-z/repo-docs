# Ethereum Proof-of-Stake Consensus Specifications: The Definitive Guide

**Dive into the heart of Ethereum's future with the official specifications for its Proof-of-Stake (PoS) consensus mechanism.** Explore the details of how Ethereum achieves secure and decentralized operations.

[View the original repository on GitHub](https://github.com/ethereum/consensus-specs)

## Key Features & Benefits

*   **Comprehensive Specifications:** Access detailed specifications, ensuring all clients implement and interact correctly.
*   **Evolving Standards:**  Stay up-to-date with the latest advancements in Ethereum's consensus layer, including planned upgrades and features.
*   **Clear Documentation:** Find resources on the architecture, design goals, and detailed specifications for understanding the inner workings of the Ethereum network.
*   **Open Community:** Participate in discussions and contribute to the evolution of the specifications through issues and pull requests.
*   **Stable and In-Development Specs:** Access the stable specifications alongside in-development specifications.
*   **Extensive Testing:** View comprehensive tests for each version of the specs.
*   **Reference Tests:** Evaluate the consensus mechanism with reference tests.

## Specifications Breakdown

The specifications are divided into core features and consolidated into sequential upgrades.

### Stable Specifications

| Seq. | Code Name     | Fork Epoch | Links                                                                        |
| ---- | ------------- | ---------- | ---------------------------------------------------------------------------- |
| 0    | **Phase0**    | `0`        | [Specs](specs/phase0), [Tests](tests/core/pyspec/eth2spec/test/phase0)       |
| 1    | **Altair**    | `74240`    | [Specs](specs/altair), [Tests](tests/core/pyspec/eth2spec/test/altair)       |
| 2    | **Bellatrix** | `144896`   | [Specs](specs/bellatrix), [Tests](tests/core/pyspec/eth2spec/test/bellatrix) |
| 3    | **Capella**   | `194048`   | [Specs](specs/capella), [Tests](tests/core/pyspec/eth2spec/test/capella)     |
| 4    | **Deneb**     | `269568`   | [Specs](specs/deneb), [Tests](tests/core/pyspec/eth2spec/test/deneb)         |
| 5    | **Electra**   | `364032`   | [Specs](specs/electra), [Tests](tests/core/pyspec/eth2spec/test/electra)     |

### In-Development Specifications

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
*   [Compressed tarballs are available for each release](https://github.com/ethereum/consensus-spec-tests/releases)

## Getting Started

### Installation

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

*   **Minimize Complexity:** Aim for simplicity, even with some efficiency trade-offs.
*   **Network Resilience:** Ensure network liveness during partitions or node failures.
*   **Quantum Security:** Choose components that are quantum-secure or easily replaceable with quantum-secure alternatives.
*   **Validator Participation:** Enable a large number of validators.
*   **Low Hardware Requirements:** Ensure consumer laptops can participate.

## Useful Resources

*   [Design Rationale](https://notes.ethereum.org/s/rkhCgQteN#)
*   [Phase0 Onboarding Document](https://notes.ethereum.org/s/Bkn3zpwxB)
*   [Combining GHOST and Casper paper](https://arxiv.org/abs/2003.03052)
*   [Specifications viewer (mkdocs)](https://ethereum.github.io/consensus-specs/)
*   [Specifications viewer (jtraglia)](https://jtraglia.github.io/eth-spec-viewer/)
*   [The Eth2 Book](https://eth2book.info)
*   [PySpec Tests](tests/core/pyspec/README.md)