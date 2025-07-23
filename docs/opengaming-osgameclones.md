# OSGameClones: Your One-Stop Resource for Open Source Game Clones and Remakes

Discover and explore a comprehensive database of open-source game clones and remakes, meticulously curated for enthusiasts and developers alike.

[![Build and Deploy](https://github.com/opengaming/osgameclones/actions/workflows/main.yml/badge.svg)](https://github.com/opengaming/osgameclones/actions/workflows/main.yml)

**Explore the OSGameClones project on GitHub: [https://github.com/opengaming/osgameclones](https://github.com/opengaming/osgameclones)**

## Key Features

*   **Extensive Game Database:** Browse a curated collection of open-source game clones and remakes, complete with detailed information.
*   **Easy Contribution:** Contribute to the database by adding new games or improving existing entries via pull requests or issues.
*   **YAML-Based Data:** Game and original game information is stored in easily accessible YAML files for easy readability and modification.
*   **Validation:** All game entries and original game references are validated against schema files, ensuring data integrity.
*   **Automated Builds:** Project includes automated build processes and Docker support for easy deployment and testing.

## Games Database Structure

The game information is stored in YAML files under the `games` directory. Original game references are stored in the `originals` directory.  Data is sorted alphabetically for easy navigation, with the exception of ScummVM, which includes many games.

## How to Contribute

We welcome contributions!

### Adding a Game Clone/Remake

1.  Create a new issue and use the [game form](https://osgameclones.com/add_game.html) to submit the details.
2.  Alternatively, edit the YAML files directly within the `games` directory and submit a pull request.  All new games must pass validation against the rules in `schema/games.yaml`.

### Adding a Reference to the Original Game

1.  Use the [add original form](https://osgameclones.com/add_original.html) to create an entry.
2.  If there is no existing game entry in `originals`, you can create a new entry following the specified format.  All original game entries are validated against `schema/originals.yaml`.

## Development & Deployment

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

### Building

Build the project into the `_build` directory:

```bash
make
```

### Running with Docker

1.  Build the Docker image:

    ```bash
    make docker-build
    ```

2.  Run the server with Docker:

    ```bash
    make docker-run
    ```

    The server will be accessible at `http://localhost:80`.  You can customize the port using the `PORT` variable:

    ```bash
    make docker-run PORT=3000 # Access at http://localhost:3000
    ```

## License

This project is licensed under the [LICENSE](LICENSE) terms.