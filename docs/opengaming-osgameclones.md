# OSGameClones: Explore Open-Source Game Clones and Remakes

**Discover a comprehensive database of open-source game clones and remakes, preserving and celebrating the history of gaming!** Explore the [OSGameClones](https://osgameclones.com/) project for a curated collection of reimaginings of classic games.

[![Build and Deploy](https://github.com/opengaming/osgameclones/actions/workflows/main.yml/badge.svg)](https://github.com/opengaming/osgameclones/actions/workflows/main.yml)

This repository powers the [OSGameClones](https://osgameclones.com/) website, a valuable resource for finding and learning about open-source game projects. Contribute to the project by adding new games or improving existing entries!

## Key Features

*   **Extensive Database:** Browse a vast collection of game clones and remakes, meticulously organized and categorized.
*   **Open Source:** All data and contributions are welcome!
*   **Easy Contribution:** Submit new games or enhance existing information via pull requests or by opening issues.
*   **Structured Data:** Game and original game information is stored in easy-to-understand YAML files, located in the [`games` directory](games/) and [`originals` directory](originals/).
*   **Validation:** Data is validated against schemas (`schema/games.yaml` and `schema/originals.yaml`) ensuring data consistency and quality.
*   **Community Driven:** Benefit from the collective knowledge and contributions of a vibrant community of game enthusiasts.

## How to Contribute

We encourage contributions from the community! Here's how you can get involved:

### Adding a Game Clone/Remake

1.  Fill out the [game form](https://osgameclones.com/add_game.html) to submit information about a new clone or remake.
2.  **Alternatively:** Edit the YAML files in the [`games` directory](games/) directly.  Your changes will be submitted as a pull request.  Ensure your changes adhere to the rules defined in [`schema/games.yaml`](schema/games.yaml).

### Adding a Reference to the Original Game

1.  Fill out the [add original form](https://osgameclones.com/add_original.html) to add information about the original game.
2.  **Alternatively:**  Create a new entry in the [`originals` directory](originals/) if it doesn't already exist, following the specified format.  Ensure your changes adhere to the rules defined in [`schema/originals.yaml`](schema/originals.yaml).

## Development & Setup

### Prerequisites

*   [poetry](https://python-poetry.org/)

### Installation

1.  Clone the repository:

    ```bash
    git clone <repository_url> # Replace with the actual repository URL if needed
    cd osgameclones
    ```
2.  Install dependencies:

    ```bash
    poetry install
    ```

### Building

Build the project to the `_build` directory:

```bash
make
```

### Running the Server with Docker

1.  Build the Docker image:

    ```bash
    make docker-build
    ```
2.  Run the server:

    ```bash
    make docker-run
    ```

    The server will be accessible at `http://localhost:80` by default. Customize the port using the `PORT` variable:

    ```bash
    make docker-run PORT=3000 # The server will be available at http://localhost:3000
    ```

## License

See the [LICENSE][license] file for details.