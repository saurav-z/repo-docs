# Open Source Game Clones: Discover and Contribute to Awesome Game Remakes & Recreations

[![Build and Deploy](https://github.com/opengaming/osgameclones/actions/workflows/main.yml/badge.svg)](https://github.com/opengaming/osgameclones/actions/workflows/main.yml)

This repository powers [osgameclones.com](https://osgameclones.com), a comprehensive database showcasing open-source game clones and remakes.  Help build and improve the ultimate resource for freely available gaming experiences!

## Key Features:

*   **Extensive Game Database:** Explore a curated collection of open-source game clones, remakes, and reimplementations.
*   **YAML-Based Data:** Game and original game information are stored in easily accessible YAML files, located in the `games/` and `originals/` directories.
*   **Easy Contribution:** Contribute new games or improve existing entries via pull requests or issues.
*   **Validation:** All game and original entries are validated against defined schemas for data integrity.
*   **Alphabetical Sorting:** Games are organized alphabetically for easy browsing (except for ScummVM which is listed separately).

## Adding and Contributing Games

### Adding a Game Clone or Remake:

1.  Use the [game form](https://osgameclones.com/add_game.html) to submit a new game entry.
2.  Alternatively, directly edit the YAML files in the `games/` directory and submit a pull request.
3.  All entries are validated against the rules defined in `schema/games.yaml`.

### Adding a Reference to the Original Game:

1.  Use the [add original form](https://osgameclones.com/add_original.html) to add a reference to the original game.
2.  If no entry exists in the `originals/` directory, create a new entry following the specified format.
3.  Original game entries are validated against `schema/originals.yaml`.

## Getting Started (Development)

### Prerequisites:

*   [poetry](https://python-poetry.org/)

### Installation:

1.  Clone this repository: `git clone [your repository URL]` (replace with the actual URL)
2.  Navigate into the project directory: `cd osgameclones`
3.  Install dependencies: `poetry install`

### Building the Project:

Run the following command to build the project into the `_build` directory:

```bash
make
```

### Running the Server with Docker:

1.  Build the Docker image:

    ```bash
    make docker-build
    ```

2.  Run the server using Docker:

    ```bash
    make docker-run
    ```

    The server will be accessible at http://localhost:80 by default.  You can customize the port using the `PORT` variable:

    ```bash
    make docker-run PORT=3000  # Server will run on http://localhost:3000
    ```

## License

See [LICENSE](LICENSE) for licensing information.

**[Back to the Original Repository](https://github.com/opengaming/osgameclones)**