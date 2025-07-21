# OSGameClones: Explore & Contribute to Open-Source Game Clones

Discover and contribute to a vast database of open-source game clones and remakes with OSGameClones!  (See the original repository at [https://github.com/opengaming/osgameclones](https://github.com/opengaming/osgameclones).)

## Key Features:

*   **Comprehensive Database:** Explore a curated collection of open-source game clones and remakes.
*   **Easy Contribution:**  Help expand the database by submitting new game entries or improving existing information.
*   **YAML-Based Data:**  Game and original game data are stored in easily accessible YAML files.
*   **Validation:**  Game entries are validated against a schema to ensure data integrity.
*   **Built-in Building and Running:**  Use simple commands to build and run the project, including Docker support.

## Games Database Structure

The core of OSGameClones is the data itself, stored in easily accessible YAML files:

*   **`games/`:** Contains information about the open-source clones and remakes.
*   **`originals/`:** References to the original games that these clones are based on.

The data is sorted alphabetically, with the exception of ScummVM which has a large number of entries.

## Contribute to the Project

We welcome contributions! Here's how you can help:

### Adding a Game Clone/Remake

1.  Fill out the [game form](https://osgameclones.com/add_game.html) to add a new game entry.
2.  Alternatively, directly edit the YAML files in the `games/` directory and submit a pull request.
3.  All game entries are validated against the rules defined in [`schema/games.yaml`](schema/games.yaml).

### Adding a Reference to the Original Game

1.  Fill out the [add original form](https://osgameclones.com/add_original.html) to reference an original game.
2.  If a game entry doesn't exist in `originals/`, create a new entry using the following format.
3.  All original entries are validated against the rules in [`schema/originals.yaml`](schema/originals.yaml).

## Development & Build Instructions

### Prerequisites

*   [poetry](https://python-poetry.org/)

### Installation

1.  Clone the repository.
2.  Navigate to the project directory in your terminal.
3.  Run `poetry install` to install dependencies.

### Building

Run the following command to build the project into the `_build` directory:

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

    The server will be available at http://localhost:80.  You can customize the port using the `PORT` variable:

    ```bash
    make docker-run PORT=3000  # Server available at http://localhost:3000
    ```

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file.