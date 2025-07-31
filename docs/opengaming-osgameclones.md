# OSGameClones: Your Comprehensive Directory of Open-Source Game Clones and Remakes

Discover a vast collection of open-source game clones and remakes, all in one place! This repository ([original repo](https://github.com/opengaming/osgameclones)) provides the source code and database behind [osgameclones.com](https://osgameclones.com), a curated list of games inspired by classic titles.

[![Build and Deploy](https://github.com/opengaming/osgameclones/actions/workflows/main.yml/badge.svg)](https://github.com/opengaming/osgameclones/actions/workflows/main.yml)

## Key Features

*   **Extensive Database:** Explore a curated list of game clones and remakes.
*   **Open-Source & Community Driven:** Contribute new games or improve existing entries via pull requests.
*   **YAML-Based Data:** Game and original game information is stored in easily readable YAML files under the `games` and `originals` directories.
*   **Validation:** All entries are validated against schema rules for data integrity.
*   **Easy Contribution:** Add new games or references using provided forms.

## Database Structure

The database of games and their corresponding original games is meticulously organized:

*   **`games/`:** Contains YAML files describing the clone/remake games.
*   **`originals/`:** Contains YAML files with information about the original games that these clones are based on.

## Contributing

We welcome contributions to expand and improve the OSGameClones database!

### Adding a Game Clone/Remake

1.  Use the [game form](https://osgameclones.com/add_game.html) to submit details of the new game.
2.  Alternatively, create a new entry by directly editing the files in the `games/` directory.
3.  Your changes will be submitted as a pull request.

### Adding a Reference to an Original Game

1.  Use the [add original form](https://osgameclones.com/add_original.html).
2.  If the original game doesn't exist, create a new entry in the `originals/` directory using the provided format.

## Prerequisites

*   [poetry](https://python-poetry.org/)

## Installation

To get started:

1.  Clone this repository.
2.  Run `poetry install` inside the project directory.

## Building

Build the project by running:

```bash
make
```

The built project will be available in the `_build` directory.

## Running with Docker

1.  Build the Docker image:

    ```bash
    make docker-build
    ```

2.  Run the server using Docker:

    ```bash
    make docker-run
    ```

    The server will be accessible at http://localhost:80. You can customize the port using the `PORT` variable:

    ```bash
    make docker-run PORT=3000  # Access at http://localhost:3000
    ```

## License

This project is licensed under the [LICENSE](LICENSE).