# OSGameClones: Discover and Contribute to Open Source Game Clones & Remakes

Explore a comprehensive database of open-source game clones and remakes, and help us grow the collection!  This project powers [osgameclones.com](https://osgameclones.com) and welcomes contributions to document and celebrate the world of open-source gaming.

[![Build and Deploy](https://github.com/opengaming/osgameclones/actions/workflows/main.yml/badge.svg)](https://github.com/opengaming/osgameclones/actions/workflows/main.yml)

## Key Features

*   **Extensive Database:** Browse a curated collection of open-source game clones and remakes.
*   **Community-Driven:** Contribute by adding new games, improving existing entries, and providing valuable information.
*   **YAML-Based Data:** Game and original game information is stored in easily accessible and editable YAML files.
*   **Validation:**  Data is validated against defined schemas to ensure consistency and accuracy.
*   **Easy Contribution:**  Contribute through pull requests or by submitting new game or original game information via forms.

## Games Database Structure

The core of the project resides in the `games/` and `originals/` directories.

*   `games/`: Contains YAML files detailing the clones and remakes.
*   `originals/`: Contains YAML files documenting the original games.

The data is structured for easy understanding and modification.  Games are sorted alphabetically, with the exception of ScummVM.

## How to Contribute

We welcome contributions of all kinds!  You can help by:

*   **Adding a New Game Clone/Remake:**  Submit a new game using the [game form](https://osgameclones.com/add_game.html) or directly edit the YAML files in the `games/` directory.
*   **Adding a Reference to an Original Game:** Use the [add original form](https://osgameclones.com/add_original.html) to link clones/remakes to their original counterparts.
*   **Improving Existing Entries:** Suggest edits or additions to improve the data quality.

All game entries are validated against the rules defined in `schema/games.yaml`, and original game entries are validated against `schema/originals.yaml`.

## Development Setup

### Prerequisites

*   [Poetry](https://python-poetry.org/)

### Installation

1.  Clone the repository:
    ```bash
    git clone <repository_url>
    cd osgameclones
    ```
2.  Install dependencies:
    ```bash
    poetry install
    ```

### Building the Project

To build the project into the `_build` directory, run:

```bash
make
```

### Running with Docker

1.  Build the Docker image:
    ```bash
    make docker-build
    ```
2.  Run the server:
    ```bash
    make docker-run
    ```
    The server will be available at `http://localhost:80`. You can change the port:
    ```bash
    make docker-run PORT=3000 # Server at http://localhost:3000
    ```

## License

This project is licensed under the [LICENSE](LICENSE) file.

##  Useful Links

*   [Project Repository](https://github.com/opengaming/osgameclones)
*   [Games Directory](games/)
*   [Originals Directory](originals/)
*   [Schema for Games](schema/games.yaml)
*   [Schema for Originals](schema/originals.yaml)
*   [Game Form](https://osgameclones.com/add_game.html)
*   [Original Game Form](https://osgameclones.com/add_original.html)