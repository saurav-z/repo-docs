# Discover & Play Open Source Game Clones!

This project curates a comprehensive database of open-source game clones and remakes, offering a fantastic resource for gamers and developers alike. Explore the [osgameclones.com](https://osgameclones.com) website and contribute to the community! [(Original Repository)](https://github.com/opengaming/osgameclones)

[![Build and Deploy](https://github.com/opengaming/osgameclones/actions/workflows/main.yml/badge.svg)](https://github.com/opengaming/osgameclones/actions/workflows/main.yml)

## Key Features:

*   **Comprehensive Game Database:**  A curated collection of open-source game clones, remakes, and reimplementations.
*   **Easy Contribution:**  Help expand the database by adding new games or updating existing information.
*   **YAML-Based Data:** Game and original game information stored in easily readable and modifiable YAML files.
*   **Data Validation:**  Games and original game entries are validated against schemas for data integrity.
*   **Community-Driven:**  Leverage the power of the open-source community to keep the database up-to-date.
*   **Website Integration**: The data from this project powers the [osgameclones.com](https://osgameclones.com) website.

## How to Contribute

We welcome contributions from the community!  Here's how you can help:

*   **Add New Games:** Use the [game form](https://osgameclones.com/add_game.html) or directly edit the YAML files in the [`games`](games/) directory.
*   **Add Original Game References:** Use the [add original form](https://osgameclones.com/add_original.html) or create entries in the [`originals`](originals/) directory.
*   **Improve Existing Data:**  Submit pull requests to correct inaccuracies or add more details.

## Technical Details

### Prerequisites

*   [poetry](https://python-poetry.org/)

### Installation

1.  Clone the repository:
2.  Navigate into the project directory.
3.  Run `poetry install`.

### Building

To build the project:

```bash
make
```
The build output is placed into the `_build` directory.

### Running with Docker

1.  Build the Docker image:

    ```bash
    make docker-build
    ```
2.  Run the server with Docker:

    ```bash
    make docker-run
    ```

    The server will be accessible at `http://localhost:80`. You can change the port using the `PORT` variable, for example:

    ```bash
    make docker-run PORT=3000
    ```
    (Server available at http://localhost:3000)

## License

See the [LICENSE][license] file for details.

[games]: games/
[originals]: originals/
[schema_games]: schema/games.yaml
[schema_originals]: schema/originals.yaml
[game_form]: https://osgameclones.com/add_game.html
[original_form]: https://osgameclones.com/add_original.html
[license]: LICENSE
[poetry]: https://python-poetry.org/