# OSGameClones: Your Ultimate Guide to Open Source Game Clones

Discover a vast and ever-growing collection of open-source game clones and remakes, all in one convenient and accessible location! ([View the original repository](https://github.com/opengaming/osgameclones))

## Key Features

*   **Comprehensive Database:** Explore a meticulously curated database of game clones, remakes, and their original counterparts.
*   **Easy Contribution:**  Contribute new games or improve existing entries through pull requests or by opening issues.
*   **YAML-Based Data:**  All game and original game data are stored in easily readable and editable YAML files.
*   **Validation:** Data is rigorously validated against schema files to ensure data quality and consistency.
*   **Web Interface:**  Enjoy a user-friendly web interface for browsing and discovering game clones: [https://osgameclones.com](https://osgameclones.com)
*   **Docker Support:**  Easily run the project locally using Docker for development and testing.

## How to Contribute

We welcome contributions from the community!  Here's how you can get involved:

### Adding a Game Clone/Remake

1.  Fill out the [game form](https://osgameclones.com/add_game.html) to suggest a new game.
2.  Alternatively, directly edit the YAML files in the [`games` directory](games/) and submit a pull request.

### Adding a Reference to the Original Game

1.  Use the [add original form](https://osgameclones.com/add_original.html) to add a reference for an original game.
2.  If the original game isn't already listed, create a new entry in the [`originals` directory](originals/).

### Prerequisites

*   [Poetry](https://python-poetry.org/) (for dependency management)

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/opengaming/osgameclones.git
    cd osgameclones
    ```
2.  Install dependencies:
    ```bash
    poetry install
    ```

### Building

Build the project into the `_build` directory:

```bash
make
```

### Running with Docker

1.  Build the Docker image:
    ```bash
    make docker-build
    ```
2.  Run the server using Docker:
    ```bash
    make docker-run
    ```
    The server will be available at http://localhost:80 by default.  You can change the port:

    ```bash
    make docker-run PORT=3000  # Server will be available at http://localhost:3000
    ```

## Data Structure & Validation

*   **`games/`**: Contains YAML files describing the game clones/remakes.
*   **`originals/`**: Contains YAML files describing the original games.
*   **`schema/games.yaml`**:  Validation schema for game entries.
*   **`schema/originals.yaml`**: Validation schema for original game entries.

## License

This project is licensed under the terms of the [LICENSE](LICENSE).