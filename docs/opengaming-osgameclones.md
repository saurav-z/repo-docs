# Explore Open Source Game Clones: Relive Classic Games!

[![Build and Deploy](https://github.com/opengaming/osgameclones/actions/workflows/main.yml/badge.svg)](https://github.com/opengaming/osgameclones/actions/workflows/main.yml)

Discover a curated collection of open-source game clones, remakes, and reimplementations, perfect for gamers and developers alike. This repository powers [osgameclones.com](https://osgameclones.com), offering a comprehensive database of games and their origins.  Contribute to the project by adding new games or enhancing existing information!

## Key Features:

*   **Extensive Game Database:**  Browse a wide variety of open-source game clones, meticulously organized and documented.
*   **YAML-Based Data:** Game information and references to original games are stored in easy-to-understand YAML files within the `games/` and `originals/` directories.
*   **Community-Driven:**  Contribute new games, improve existing entries, and help grow the database through pull requests and issues.
*   **Validation:**  Ensure data integrity with validation rules defined in `schema/games.yaml` and `schema/originals.yaml`.
*   **Easy to Contribute:**  Utilize user-friendly forms on [osgameclones.com](https://osgameclones.com) to add games or original game references or edit files directly.
*   **Docker Support:** Easily run the project with Docker for a quick and consistent development environment.

##  Contributing

We welcome contributions! Here's how to get started:

### Prerequisites

*   [Poetry](https://python-poetry.org/)

### Installation

1.  Clone the repository:

    ```bash
    git clone [YOUR REPO URL HERE]
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

### Running with Docker

1.  Build the Docker image:

    ```bash
    make docker-build
    ```

2.  Run the server with Docker:

    ```bash
    make docker-run
    ```

    The server will be available at http://localhost:80.  You can customize the port using the `PORT` variable:

    ```bash
    # The server will be available at http://localhost:3000
    make docker-run PORT=3000
    ```

## Adding Games and References

### Adding a Clone/Remake

Use the [game form](https://osgameclones.com/add_game.html) or directly edit files in the `games/` directory. Your changes will be submitted as a pull request. All games will be validated against the rules in `schema/games.yaml`.

### Adding a Reference to the Original Game

Use the [add original form](https://osgameclones.com/add_original.html). If the original game isn't already listed, create a new entry in the `originals/` directory following the specified format. All originals are validated against the rules in `schema/originals.yaml`.

## License

See [LICENSE](LICENSE)

**[View the original repository on GitHub](https://github.com/opengaming/osgameclones)**