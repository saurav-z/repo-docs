# Open Source Game Clones: A Comprehensive Database (osgameclones.com)

Discover and explore a vast collection of open-source clones and remakes of classic video games with [osgameclones.com](https://osgameclones.com), a community-driven resource for retro gaming enthusiasts.

[![Build and Deploy](https://github.com/opengaming/osgameclones/actions/workflows/main.yml/badge.svg)](https://github.com/opengaming/osgameclones/actions/workflows/main.yml)

This repository powers the [osgameclones.com](https://osgameclones.com) website. Contribute to the project by adding new games, improving existing information, and helping the community!

## Key Features

*   **Extensive Database:** Explore a growing database of open-source game clones and remakes, meticulously curated.
*   **Community-Driven:** Contribute to the project by submitting pull requests or opening issues.
*   **YAML-Based Data:** Game and original game information is stored in easy-to-understand YAML files.
*   **Validation:** Games and original game entries are validated against schema files to maintain data integrity.
*   **Easy Contribution:** Contribute new games or original game references through issue forms or by editing the YAML files directly.

## How It Works

The website and its data are powered by:

*   **`games/` Directory:** Contains YAML files describing the open-source clones/remakes.
*   **`originals/` Directory:** Contains YAML files referencing the original games.
*   **Schema Files:** `schema/games.yaml` and `schema/originals.yaml` are used for data validation.

## Contribute

We welcome contributions! Here's how to get involved:

*   **Add a Clone/Remake:** Fill out the [game form](https://osgameclones.com/add_game.html) to submit a new game.  Alternatively, edit the YAML files in the [`games`][games] directory directly.
*   **Add an Original Game Reference:** Use the [add original form](https://osgameclones.com/add_original.html) to add a reference to the original game. Or, create a new entry in the `originals/` directory.

## Development

### Prerequisites

*   [poetry][poetry]

### Installation

Clone the repository and run:

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

2.  Run the server:

    ```bash
    make docker-run
    ```

    The server will be available at http://localhost:80.  You can change the port using the `PORT` variable:

    ```bash
    make docker-run PORT=3000
    ```

    (This will run the server on http://localhost:3000)

## License

See the [LICENSE][license] file.

[games]: games/
[originals]: originals/
[schema_games]: schema/games.yaml
[schema_originals]: schema/originals.yaml
[game_form]: https://osgameclones.com/add_game.html
[original_form]: https://osgameclones.com/add_original.html
[license]: LICENSE
[python]: https://www.python.org
[poetry]: https://python-poetry.org/