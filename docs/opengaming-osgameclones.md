# OSGameClones: Explore Open Source Game Clones & Remakes

Discover a comprehensive database of open-source game clones and remakes, allowing you to revisit classic games and explore their implementations. Check out the official site at [https://osgameclones.com](https://osgameclones.com) and contribute to the project on [GitHub](https://github.com/opengaming/osgameclones).

[![Build and Deploy](https://github.com/opengaming/osgameclones/actions/workflows/main.yml/badge.svg)](https://github.com/opengaming/osgameclones/actions/workflows/main.yml)

## Key Features:

*   **Extensive Game Database:**  Browse a curated collection of open-source game clones and remakes, meticulously categorized.
*   **Easy Contribution:** Add new games or enhance existing entries by submitting pull requests or opening issues.
*   **YAML-Based Data:** Game information and original game references are stored in easy-to-understand YAML files within the `games` and `originals` directories, simplifying data access and modification.
*   **Data Validation:**  All game and original entries are validated against schema files (`schema/games.yaml` and `schema/originals.yaml`) to ensure data integrity.
*   **Community Driven:**  Join a community of gaming enthusiasts and contribute to a living resource for open-source gaming.

## Contributing to OSGameClones

Want to contribute? We welcome your contributions!

### Adding a Game Clone/Remake

To add a new game clone or remake:

1.  Fill out the [game form](https://osgameclones.com/add_game.html) on the website.
2.  Alternatively, edit the YAML files directly in the `games` directory.  Your changes will be submitted as a pull request.

### Adding a Reference to the Original Game

To add a reference to the original game:

1.  Fill out the [add original form](https://osgameclones.com/add_original.html).
2.  If the original game doesn't exist, create a new entry in the `originals` directory.

## Development & Building

### Prerequisites
*   [poetry](https://python-poetry.org/)

### Installation

1.  Clone the repository.
2.  Navigate to the project directory.
3.  Run:

    ```bash
    poetry install
    ```

### Building

To build the project into the `_build` directory:

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

    The server will be available at `http://localhost:80`.  You can customize the port using the `PORT` variable:

    ```bash
    make docker-run PORT=3000
    ```
    (Server will be available at `http://localhost:3000`)

## License

See [LICENSE](LICENSE)