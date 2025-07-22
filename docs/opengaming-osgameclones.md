# Open Source Game Clones: Discover and Contribute to Classic Game Remakes

[![Build and Deploy](https://github.com/opengaming/osgameclones/actions/workflows/main.yml/badge.svg)](https://github.com/opengaming/osgameclones/actions/workflows/main.yml)

This repository powers [osgameclones.com](https://osgameclones.com), a comprehensive directory of open-source game clones and remakes.  Contribute to the project by adding new games, improving existing entries, and helping preserve the legacy of classic gaming!

## Key Features

*   **Extensive Database:** Browse a curated list of open-source game clones, remakes, and reimplementations.
*   **Easy Contribution:**  Submit new games or updates via pull requests or by creating issues.
*   **YAML-Based Data:**  Game information is stored in easily readable YAML files, simplifying contributions and modifications.
*   **Validation:**  All game and original game entries are validated against schema files to ensure data integrity.
*   **Docker Support:** Quickly set up and run the project with Docker.

## Games Database Structure

The project's data is organized for easy navigation and contribution:

*   **`games/`**:  Contains YAML files, each detailing a specific game clone or remake.
*   **`originals/`**:  Stores information about the original games that the clones are based on.
*   **Alphabetical Sorting:** Games are sorted alphabetically for easy browsing, with ScummVM exceptions.
*   **Schema Validation:** Game and original entries are validated against the `schema/games.yaml` and `schema/originals.yaml` files, respectively.

## How to Contribute

We welcome contributions from the community! Here's how you can get involved:

### Adding a Game Clone or Remake

1.  Use the [game form][game_form] to submit a new game.
2.  Alternatively, edit the YAML files directly in the `games/` directory and submit a pull request.
3.  Ensure your game entry adheres to the validation rules defined in `schema/games.yaml`.

### Adding a Reference to the Original Game

1.  Use the [add original form][original_form] to submit information about the original game.
2.  If the original game is not already listed, create a new entry in the `originals/` directory following the existing format.
3.  Ensure your original game entry adheres to the validation rules defined in `schema/originals.yaml`.

### Pre-requisites

*   [poetry][poetry]

### Installation

1.  Clone this repository:
    ```bash
    git clone https://github.com/opengaming/osgameclones.git
    cd osgameclones
    ```

2.  Install dependencies using Poetry:
    ```bash
    poetry install
    ```

### Building the Project

Build the project into the `_build` directory:

```bash
make
```

### Running the Server with Docker

1.  Build the Docker image:

    ```bash
    make docker-build
    ```

2.  Run the server with Docker:

    ```bash
    make docker-run
    ```

    The server will be available at http://localhost:80 by default.  You can change the port using the `PORT` variable:

    ```bash
    make docker-run PORT=3000  # Server will be available at http://localhost:3000
    ```

## License

See the [LICENSE][license] file for licensing information.

**[Visit the original repository on GitHub](https://github.com/opengaming/osgameclones) to learn more and contribute!**

[games]: games/
[originals]: originals/
[schema_games]: schema/games.yaml
[schema_originals]: schema/originals.yaml
[game_form]: https://osgameclones.com/add_game.html
[original_form]: https://osgameclones.com/add_original.html
[license]: LICENSE

[python]: https://www.python.org
[poetry]: https://python-poetry.org/