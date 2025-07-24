# Discover & Contribute to Open Source Game Clones

[![Build and Deploy](https://github.com/opengaming/osgameclones/actions/workflows/main.yml/badge.svg)](https://github.com/opengaming/osgameclones/actions/workflows/main.yml)

Explore and contribute to a comprehensive database of open-source game clones and remakes! This repository powers [osgameclones.com](https://osgameclones.com), providing a curated collection of reimplemented classic games.

## Key Features

*   **Extensive Database:** Browse a curated list of open-source game clones, including their original game references.
*   **Community-Driven:**  Contribute new game clones or improve existing entries through pull requests and issue submissions.
*   **YAML-Based Data:** Game and original game data are stored in easily-readable YAML files within the `games` and `originals` directories.
*   **Validation:**  Data is validated against schemas (`schema/games.yaml` and `schema/originals.yaml`) ensuring data integrity.
*   **Easy Contribution:**  Add new games using the [game form](https://osgameclones.com/add_game.html) or by directly editing the YAML files.

## Contributing

Help us grow the database of open source game clones!  Here's how you can contribute:

### Prerequisites

*   [Poetry](https://python-poetry.org/) (for dependency management)

### Installation

1.  Clone the repository:

    ```bash
    git clone <repository_url>  # Replace with the actual repository URL
    cd osgameclones
    ```
2.  Install dependencies:

    ```bash
    poetry install
    ```

### Building the Project

To build the project:

```bash
make
```

### Running the Server with Docker

1.  Build the Docker image:

    ```bash
    make docker-build
    ```
2.  Run the server using Docker:

    ```bash
    make docker-run
    ```

    The server will be accessible at `http://localhost:80` (or the port specified by the `PORT` variable).

    ```bash
    # Example: Run on port 3000
    make docker-run PORT=3000
    ```

## License

This project is licensed under the [LICENSE][license] (replace with the actual license if different).

[games]: games/
[originals]: originals/
[schema_games]: schema/games.yaml
[schema_originals]: schema/originals.yaml
[game_form]: https://osgameclones.com/add_game.html
[original_form]: https://osgameclones.com/add_original.html
[license]: LICENSE

[python]: https://www.python.org
[poetry]: https://python-poetry.org/