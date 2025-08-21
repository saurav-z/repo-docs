# Discover Open Source Game Clones: Relive Classic Gaming Experiences

[![Build and Deploy](https://github.com/opengaming/osgameclones/actions/workflows/main.yml/badge.svg)](https://github.com/opengaming/osgameclones/actions/workflows/main.yml)

Explore a curated collection of open-source game clones and remakes, bringing classic gaming experiences to life! This repository powers [osgameclones.com](https://osgameclones.com), your go-to resource for discovering and contributing to open-source gaming projects.

## Key Features

*   **Comprehensive Database:** A growing database of open-source game clones, remakes, and related projects.
*   **Detailed Game Information:** Each entry includes references to the original games, providing context and background.
*   **Easy Contribution:** Contribute new game clones or improve existing information through pull requests and issue submissions.
*   **YAML-Based Data:** Game and original game data is stored in human-readable YAML files for easy understanding and modification.
*   **Data Validation:**  Game entries and original game entries are validated against schemas for data integrity.
*   **Docker Support:**  Easily run the project with Docker for convenient local development and deployment.

## Games Database & Original Games

The core of the project resides in the `games/` and `originals/` directories. These directories contain YAML files that define the games and their connections to original titles. The data is alphabetically sorted, with the exception of ScummVM for convenience.

*   **`games/`:** Contains information about the open-source game clones.
*   **`originals/`:** Contains information about the original games that the clones are based on.

## Contribute to the Collection

We welcome contributions from the community!  Help us expand and improve the database.

### Adding a Game Clone/Remake

1.  **Create a new issue** using the [game form](https://osgameclones.com/add_game.html) to submit game details.
2.  **Directly edit YAML files:** You can also contribute directly by editing the YAML files within the `games/` directory, and submitting a pull request.

All game entries are validated against the rules defined in the [`schema/games.yaml`](schema/games.yaml) file.

### Adding a Reference to an Original Game

1.  **Use the [add original form](https://osgameclones.com/add_original.html).**
2.  If the original game does not exist in the `originals/` directory, create a new entry.

All original game entries are validated against the rules defined in the [`schema/originals.yaml`](schema/originals.yaml) file.

## Getting Started (Local Development)

### Prerequisites

*   [Poetry](https://python-poetry.org/) (for dependency management)

### Installation

1.  Clone the repository:

    ```bash
    git clone [repository URL]
    cd osgameclones
    ```
2.  Install dependencies:

    ```bash
    poetry install
    ```

### Building the Project

Run the following command to build the project into the `_build` directory:

```bash
make
```

### Running with Docker

1.  Build the Docker image:

    ```bash
    make docker-build
    ```
2.  Run the server using Docker:

    ```bash
    make docker-run
    ```

    The server will be accessible at `http://localhost:80`.  You can customize the port using the `PORT` variable:

    ```bash
    make docker-run PORT=3000 # Server available at http://localhost:3000
    ```

## License

See the [LICENSE](LICENSE) file for details.

[Back to the original repo](https://github.com/opengaming/osgameclones)