# Explore the World of Open Source Game Clones

**Discover a vast collection of open-source game clones and remakes, meticulously documented and ready for your exploration!**  Explore the original project on [GitHub](https://github.com/opengaming/osgameclones).

[![Build and Deploy](https://github.com/opengaming/osgameclones/actions/workflows/main.yml/badge.svg)](https://github.com/opengaming/osgameclones/actions/workflows/main.yml)

## Key Features

*   **Comprehensive Database:** A curated database of open-source game clones, providing links and information about each project.
*   **Organized Data:** Game information is stored in easy-to-read YAML files, categorized under `games/` and `originals/` directories.
*   **Contribution-Friendly:** Easily contribute new games or improve existing entries through pull requests or by creating new issues.
*   **Validation:**  All game and original entries are validated against schema rules to ensure data consistency and accuracy.
*   **Built with Modern Tools:** Uses Poetry for dependency management and Make for building and running.
*   **Docker Support:** Run the project easily with Docker for quick setup and deployment.

## Contributing to the Project

Help us build the ultimate resource for open-source game clones!

### Adding a Game Clone or Remake

1.  Fill out the [game form](https://osgameclones.com/add_game.html) to submit information about a new clone.
2.  Alternatively, you can directly edit the YAML files in the `games/` directory and submit a pull request. Your changes will be validated against the rules in [`schema/games.yaml`][schema_games].

### Adding a Reference to the Original Game

1.  Use the [add original form](https://osgameclones.com/add_original.html) to add a reference to the original game.
2.  If the original game doesn't exist in the `originals/` directory, you can create a new entry following the established format. All originals are validated against the rules in [`schema/originals.yaml`][schema_originals].

## Development Setup

### Prerequisites

*   [Poetry](https://python-poetry.org/)

### Installation

1.  Clone the repository:

    ```bash
    git clone <repository_url>  # Replace with the actual repo URL
    cd osgameclones
    ```
2.  Install dependencies:

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

    The server will be available at http://localhost:80. You can customize the port using the `PORT` variable:

    ```bash
    make docker-run PORT=3000  # Server available at http://localhost:3000
    ```

## License

See the [LICENSE][license] file for licensing information.

[games]: games/
[originals]: originals/
[schema_games]: schema/games.yaml
[schema_originals]: schema/originals.yaml
[license]: LICENSE