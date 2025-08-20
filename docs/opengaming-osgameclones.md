# Explore the World of Open Source Game Clones

**Discover and contribute to a comprehensive database of open-source game clones and remakes at [https://osgameclones.com](https://osgameclones.com)!**

[![Build and Deploy](https://github.com/opengaming/osgameclones/actions/workflows/main.yml/badge.svg)](https://github.com/opengaming/osgameclones/actions/workflows/main.yml)

This project provides the source code for [osgameclones.com](https://osgameclones.com), a valuable resource for finding and exploring open-source reimplementations of classic games.

## Key Features:

*   **Extensive Database:** Access a curated collection of open-source game clones and remakes.
*   **Community Driven:** Contribute to the project by adding new games or improving existing entries via pull requests or issues.
*   **Organized Data:** Game information and references to original games are stored in easily accessible YAML files.
*   **Validation:**  Game and original game data are validated against schema files (`schema/games.yaml` and `schema/originals.yaml`) to ensure data integrity.
*   **Easy Contribution:** Add new clones/remakes and original game references using provided forms or by directly editing the data files.

## How to Contribute

We welcome contributions from the community! Here's how you can get involved:

### Adding a Game Clone/Remake

1.  Use the [game form](https://osgameclones.com/add_game.html) to submit information about a new clone.
2.  Alternatively, directly edit the files within the [`games` directory](games/) and submit a pull request.

### Adding a Reference to an Original Game

1.  Use the [add original form](https://osgameclones.com/add_original.html) to add a reference to the original game a clone is based on.
2.  If no entry exists in the [`originals` directory](originals/), create a new entry following the existing format.

## Setting up the Project

### Prerequisites

*   [Poetry](https://python-poetry.org/)

### Installation

1.  Clone the repository: `git clone https://github.com/opengaming/osgameclones.git`
2.  Navigate to the project directory: `cd osgameclones`
3.  Install dependencies: `poetry install`

### Building the Project

To build the project, run:

```bash
make
```

This will build the project into the `_build` directory.

### Running with Docker

1.  Build the Docker image:

    ```bash
    make docker-build
    ```

2.  Run the server using Docker:

    ```bash
    make docker-run
    ```

    The server will be accessible at `http://localhost:80`. You can customize the port using the `PORT` variable:

    ```bash
    # The server will be available at http://localhost:3000
    make docker-run PORT=3000
    ```

## License

This project is licensed under the [LICENSE](LICENSE)

**Explore the Open Gaming World - Visit the [original repository](https://github.com/opengaming/osgameclones) to contribute or learn more!**