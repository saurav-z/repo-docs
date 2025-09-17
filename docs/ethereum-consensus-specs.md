# Ethereum Proof-of-Stake (PoS) Consensus Specifications

**Dive into the technical blueprint of Ethereum's transition to Proof-of-Stake, ensuring network security and scalability.**  ([Original Repository](https://github.com/ethereum/consensus-specs))

This repository serves as the definitive source for the specifications that govern the Ethereum Proof-of-Stake (PoS) consensus mechanism. It provides detailed documentation, test vectors, and resources for developers and researchers interested in understanding and contributing to the future of Ethereum.

[![Join the chat on Discord](https://img.shields.io/badge/chat-on%20discord-blue.svg)](https://discord.gg/qGpsxSA)
[![Generate Test Vectors](https://github.com/ethereum/consensus-specs/actions/workflows/generate_vectors.yml/badge.svg?branch=dev&event=schedule)](https://github.com/ethereum/consensus-specs/actions/workflows/generate_vectors.yml)

## Key Features

*   **Comprehensive Specifications:** Detailed specifications for all phases and upgrades of the Ethereum PoS protocol.
*   **Versioned Releases:**  Clear separation of stable and in-development specifications, with each release defined by a code name and fork epoch.
*   **Test Vectors & Test Suites:** Robust test suites and vectors to ensure client implementations adhere to the specifications.
*   **SSZ and Merkle Proofs:**  Specifications for SimpleSerialize (SSZ) and Merkle proof formats for efficient data handling.
*   **External Specifications:** Links to related specifications like Beacon APIs, Engine APIs, and Builder Specs.
*   **Developer Resources:** Installation instructions and design goals to help you get started and understand the rationale behind the design.

## Specifications

Core specifications for Ethereum proof-of-stake clients can be found in the `specs` directory, organized by feature and upgrade.

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

Reference tests built from the executable Python spec are available in the
[Ethereum Proof-of-Stake Consensus Spec Tests](https://github.com/ethereum/consensus-spec-tests)
repository. Compressed tarballs are available for each release
[here](https://github.com/ethereum/consensus-spec-tests/releases). Nightly
reference tests are available
[here](https://github.com/ethereum/consensus-specs/actions/workflows/generate_vectors.yml).

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

View available commands:

```bash
make help
```

## Design Goals

The Ethereum PoS consensus specifications are guided by the following design goals:

*   **Minimize Complexity:** Prioritizing simplicity, even if it means some efficiency trade-offs.
*   **Network Resilience:** Ensuring the network remains live through major partitions and node failures.
*   **Quantum Security:** Incorporating quantum-resistant components or easily swappable alternatives.
*   **Validator Participation:** Employing cryptographic and design techniques to facilitate a large number of validators.
*   **Accessibility:** Minimizing hardware requirements for participation, allowing consumer laptops to validate.

## Useful Resources

*   [Design Rationale](https://notes.ethereum.org/s/rkhCgQteN#)
*   [Phase0 Onboarding Document](https://notes.ethereum.org/s/Bkn3zpwxB)
*   [Combining GHOST and Casper paper](https://arxiv.org/abs/2003.03052)
*   [Specifications viewer (mkdocs)](https://ethereum.github.io/consensus-specs/)
*   [Specifications viewer (jtraglia)](https://jtraglia.github.io/eth-spec-viewer/)
*   [The Eth2 Book](https://eth2book.info)
*   [PySpec Tests](tests/core/pyspec/README.md)