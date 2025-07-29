# OSGameClones: Explore and Contribute to Open Source Game Clones

Discover and contribute to a comprehensive database of open-source game clones and remakes, inspired by classic and popular titles. This repository powers [osgameclones.com](https://osgameclones.com).

## Key Features:

*   **Extensive Database:** Explore a curated collection of game clones, each linked to its original game.
*   **Easy Contribution:**  Contribute new games or improve existing entries through pull requests or by opening an issue.
*   **YAML-Based Data:** Game and original game data are stored in organized YAML files, making them easy to understand and modify.
*   **Validation:** All game and original game data is validated against predefined schemas for data integrity.
*   **[osgameclones.com](https://osgameclones.com):** This repository is the source of the website.

## How to Contribute

Help expand the OSGameClones database by adding new games or improving existing entries.

### Adding a Game Clone/Remake

1.  Use the [game form](https://osgameclones.com/add_game.html) to submit details about a new game.
2.  Alternatively, edit the YAML files directly within the `games` directory.
3.  Your changes will be submitted as a pull request.
4.  All game entries are validated against the rules defined in `schema/games.yaml`.

### Adding a Reference to an Original Game

1.  Use the [add original form](https://osgameclones.com/add_original.html) to provide details about the original game.
2.  If there isn't an existing entry in the `originals` directory, create a new one following the established format.
3.  All original game entries are validated against the rules defined in `schema/originals.yaml`.

## Development and Building

### Prerequisites

*   [poetry](https://python-poetry.org/)

### Installation

1.  Clone the repository.
2.  Navigate to the project directory in your terminal.
3.  Run: `poetry install`

### Building the Project

Run the following command to build the project into the `_build` directory:

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

    The server will be available at http://localhost:80 by default, but you can customize the port:

    ```bash
    make docker-run PORT=3000 # Server will be at http://localhost:3000
    ```

## License

See the [LICENSE](LICENSE) file.

---

**[Visit the original repository on GitHub](https://github.com/opengaming/osgameclones) for the latest updates and more information.**