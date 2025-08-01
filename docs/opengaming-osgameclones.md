# Open Source Game Clones: Explore & Contribute to Classic Game Remakes & Clones

Discover a curated collection of open-source game clones and remakes with [osgameclones.com](https://osgameclones.com), a community-driven resource showcasing reimplementations of classic games.  You can contribute to this project by adding new games or improving existing information.

[![Build and Deploy](https://github.com/opengaming/osgameclones/actions/workflows/main.yml/badge.svg)](https://github.com/opengaming/osgameclones/actions/workflows/main.yml)

**Key Features:**

*   **Comprehensive Database:**  Access a meticulously organized database of game clones and their original counterparts, stored in YAML files for easy management and contribution.
*   **Community-Driven:**  Contribute to the project by submitting pull requests or opening issues to add new games, improve information, and help expand the collection.
*   **Clear Structure:** Understand the game's structure by viewing the `games` and `originals` directories.
*   **Validation:** All game and original entries are validated against schemas for data integrity.

**How to Contribute:**

## Adding a New Game Clone/Remake

1.  Use the user-friendly [game form](https://osgameclones.com/add_game.html) to submit details about a new game.
2.  Alternatively, directly edit the YAML files within the [`games`][games] directory and submit a pull request.
3.  Your submissions are validated against the rules defined in [`schema/games.yaml`][schema_games].

## Adding a Reference to an Original Game

1.  Utilize the [add original form](https://osgameclones.com/add_original.html) to add a reference to the original game a clone is based on.
2.  If the original game doesn't exist in the [`originals`][originals] directory, create a new entry following the specified format.
3.  Original game entries are validated against the rules outlined in [`schema/originals.yaml`][schema_originals].

## Contributing to the Project

### Prerequisites

*   [poetry][poetry]

### Installation

1.  Clone the repository.
2.  Navigate to the project directory.
3.  Run the following command to install dependencies:

    ```bash
    poetry install
    ```

### Building the Project

To build the project into the `_build` directory, execute:

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

    The server will be accessible at http://localhost:80.  You can customize the port using the **PORT** variable:

    ```bash
    # The server will be available at http://localhost:3000
    make docker-run PORT=3000
    ```

## License

See [LICENSE][license] for licensing information.

**[View the original repository on GitHub](https://github.com/opengaming/osgameclones)**

[games]: games/
[originals]: originals/
[schema_games]: schema/games.yaml
[schema_originals]: schema/originals.yaml
[game_form]: https://osgameclones.com/add_game.html
[original_form]: https://osgameclones.com/add_original.html
[license]: LICENSE
[poetry]: https://python-poetry.org/