# Ethereum Proof-of-Stake Consensus Specifications: The Future of Ethereum

**Dive into the core specifications driving the Ethereum blockchain's transition to Proof-of-Stake, ensuring a more secure and scalable future.**  [Explore the original repository here](https://github.com/ethereum/consensus-specs).

[![Join the chat on Discord](https://img.shields.io/badge/chat-on%20discord-blue.svg)](https://discord.gg/qGpsxSA)
[![Test Generation Workflow Status](https://github.com/ethereum/consensus-specs/actions/workflows/generate_vectors.yml/badge.svg?branch=dev&event=schedule)](https://github.com/ethereum/consensus-specs/actions/workflows/generate_vectors.yml)

## Key Features

*   **Comprehensive Specifications:** Detailed specifications for Ethereum's Proof-of-Stake (PoS) consensus mechanism, including core components and upgrades.
*   **Versioned Upgrades:**  Clear separation of specifications by code name (e.g., Phase0, Altair, Capella, Deneb, Electra, Fulu, Gloas), allowing for organized research and development.
*   **Modular Design:** Specifications are divided into features, enabling parallel development and streamlined integration into sequential upgrades.
*   **Rigorous Testing:**  Includes accompanying test suites to ensure the reliability and correctness of the specifications.
*   **External Specifications:** Links to relevant specifications like Beacon APIs, Engine APIs, and Builder Specs.
*   **Active Community:**  Open to community contributions and discussions via issues and pull requests.
*   **Design Goals:** Focused on minimizing complexity, ensuring network resilience, quantum-security considerations, validator participation, and minimizing hardware requirements.

## Specifications

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

Reference tests are available [here](https://github.com/ethereum/consensus-spec-tests). Nightly reference tests are available [here](https://github.com/ethereum/consensus-specs/actions/workflows/generate_vectors.yml).

## Getting Started

### Installation

```bash
git clone https://github.com/ethereum/consensus-specs.git
cd consensus-specs
make help
```

## Design Goals

The Ethereum Proof-of-Stake consensus specifications are designed to:

*   Minimize complexity.
*   Maintain network stability during partitions and node outages.
*   Incorporate quantum-secure components or easily swappable alternatives.
*   Facilitate high validator participation.
*   Reduce hardware requirements.

## Useful Resources

*   [Design Rationale](https://notes.ethereum.org/s/rkhCgQteN#)
*   [Phase0 Onboarding Document](https://notes.ethereum.org/s/Bkn3zpwxB)
*   [Combining GHOST and Casper paper](https://arxiv.org/abs/2003.03052)
*   [Specifications viewer (mkdocs)](https://ethereum.github.io/consensus-specs/)
*   [Specifications viewer (jtraglia)](https://jtraglia.github.io/eth-spec-viewer/)
*   [The Eth2 Book](https://eth2book.info)
*   [PySpec Tests](tests/core/pyspec/README.md)