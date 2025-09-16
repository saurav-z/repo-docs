# Ethereum Proof-of-Stake (PoS) Consensus Specifications

**Dive into the core technical specifications that power the Ethereum network's proof-of-stake consensus mechanism.**  Learn about the design, development, and ongoing evolution of Ethereum's PoS, and how it contributes to a more secure, scalable, and sustainable blockchain.  For the original source of truth, see the [Ethereum Consensus Specs](https://github.com/ethereum/consensus-specs) repository.

## Key Features:

*   **Comprehensive Specifications:** Detailed documentation covering all aspects of the Ethereum proof-of-stake consensus, from core functionality to future upgrades.
*   **Up-to-Date Information:** Stay informed about the latest developments and changes through ongoing updates and revisions to the specifications.
*   **Modular Design:** Specifications are divided into features, allowing for parallel research, development, and a streamlined upgrade process.
*   **Stable & In-Development Versions:** Access both finalized and actively evolving specifications, including code names, fork epochs, and relevant links.
*   **Extensive Testing:** Validate the specifications through comprehensive tests, including reference tests and test vectors, ensuring the robustness and reliability of the PoS mechanism.
*   **SSZ and Merkle Proofs:** Understand the underlying SimpleSerialize (SSZ) and Merkle proof formats that ensure data integrity and efficient verification.
*   **External Resources:** Access additional specifications and standards, including Beacon APIs, Engine APIs, Beacon Metrics, and Builder Specs, to gain a comprehensive understanding of the Ethereum ecosystem.
*   **Developer-Friendly Resources:** Access a suite of resources including a specifications viewer, design rationale, and more to aid in your understanding and development.

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

## Reference Tests

Reference tests built from the executable Python spec are available in the
[Ethereum Proof-of-Stake Consensus Spec Tests](https://github.com/ethereum/consensus-spec-tests)
repository. Compressed tarballs are available for each release
[here](https://github.com/ethereum/consensus-spec-tests/releases). Nightly
reference tests are available
[here](https://github.com/ethereum/consensus-specs/actions/workflows/generate_vectors.yml).

## Design Goals

*   Minimize complexity.
*   Maintain liveness through network partitions and node failures.
*   Employ quantum-resistant or easily replaceable components.
*   Support large validator participation.
*   Minimize hardware requirements for participation.

## Get Started

### Installation and Usage

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/ethereum/consensus-specs.git
    ```

2.  **Navigate to the directory:**

    ```bash
    cd consensus-specs
    ```

3.  **View help output:**

    ```bash
    make help
    ```

## Useful Resources

*   [Design Rationale](https://notes.ethereum.org/s/rkhCgQteN#)
*   [Phase0 Onboarding Document](https://notes.ethereum.org/s/Bkn3zpwxB)
*   [Combining GHOST and Casper paper](https://arxiv.org/abs/2003.03052)
*   [Specifications viewer (mkdocs)](https://ethereum.github.io/consensus-specs/)
*   [Specifications viewer (jtraglia)](https://jtraglia.github.io/eth-spec-viewer/)
*   [The Eth2 Book](https://eth2book.info)
*   [PySpec Tests](tests/core/pyspec/README.md)