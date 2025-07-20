# Discover & Contribute to Open Source Game Clones

Explore a comprehensive database of open-source game clones and remakes, preserving and celebrating classic gaming experiences.  This project is the source for [osgameclones.com](https://osgameclones.com).

[![Build and Deploy](https://github.com/opengaming/osgameclones/actions/workflows/main.yml/badge.svg)](https://github.com/opengaming/osgameclones/actions/workflows/main.yml)

## Key Features:

*   **Curated Database:** A meticulously maintained collection of open-source game clones, remakes, and reimplementations.
*   **YAML-Based Data:** Game and original game information is stored in easily readable and editable YAML files, ensuring transparency and maintainability.
*   **Contribution Encouraged:**  Actively seek contributions to add new games and improve existing information, fostering community involvement.
*   **Validation:**  Data is validated against schemas (`schema/games.yaml` and `schema/originals.yaml`) to maintain data integrity.
*   **Easy to Build & Run:** Includes clear instructions for building, running, and deploying the project with Docker.

## How to Contribute

We welcome contributions to expand and improve the database of open-source game clones.

### Adding a Game Clone/Remake

1.  Fill out the [game form](https://osgameclones.com/add_game.html) to submit a new game.
2.  Alternatively, edit the YAML files directly in the [`games` directory](games/) and submit a pull request.
3.  Your changes will be validated against the rules defined in [`schema/games.yaml`](schema/games.yaml).

### Adding a Reference to the Original Game

1.  Fill out the [add original form](https://osgameclones.com/add_original.html) to add a reference to the original game.
2.  If the original game isn't already in the [`originals` directory](originals/), create a new entry, following the specified format.
3.  Original game entries are validated against the rules in [`schema/originals.yaml`](schema/originals.yaml).

## Development & Deployment

### Prerequisites

*   [poetry](https://python-poetry.org/)

### Installation

Clone the repository and run:

```bash
poetry install
```

### Building

To build the project into the `_build` directory, run:

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

    The server will be available at `http://localhost:80`.  You can customize the port using the `PORT` variable:

    ```bash
    make docker-run PORT=3000  # Server will be available at http://localhost:3000
    ```

## License

See the [LICENSE](LICENSE) file for licensing information.

**Explore the full project on GitHub: [https://github.com/opengaming/osgameclones](https://github.com/opengaming/osgameclones)**