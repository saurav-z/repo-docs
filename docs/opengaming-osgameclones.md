# Discover and Contribute to Open Source Game Clones

Explore a vast collection of open-source game clones and remakes, and help expand the database! ([View the original repository](https://github.com/opengaming/osgameclones))

## Overview

This project hosts a curated database of open-source game clones and remakes, providing information and links to these fan-made recreations of classic games. Contribute to the community by adding new games, updating information, and helping preserve gaming history. The database powers [https://osgameclones.com](https://osgameclones.com).

## Key Features

*   **Comprehensive Database:** Discover clones and remakes of various classic games.
*   **Community-Driven:**  Contribute by adding new games, updating existing entries, and improving the information.
*   **YAML-Based Data:** Games and original game information are stored in easily accessible and editable YAML files under the `games/` and `originals/` directories.
*   **Validation:**  Data is validated against schemas (`schema/games.yaml` and `schema/originals.yaml`) to ensure data integrity.
*   **Easy Contribution:**  Submit changes via pull requests or by opening issues.  Forms are provided for easy game and original game submissions.

## How to Contribute

### Adding a New Game Clone/Remake

1.  **Use the Game Form:** Submit a new game using the [game form](https://osgameclones.com/add_game.html).
2.  **Direct Editing:** Alternatively, edit the YAML files directly in the `games/` directory. Your changes will be submitted as a pull request.

### Adding a Reference to the Original Game

1.  **Use the Original Game Form:** Add a reference to the original game using the [add original form](https://osgameclones.com/add_original.html).
2.  **Create a New Entry:** If the original game isn't listed, create a new entry in the `originals/` directory, following the specified format.

## Development & Build

### Prerequisites

*   [poetry](https://python-poetry.org/)

### Installation

Clone the repository and run:

```bash
poetry install
```

### Building

To build the project into the `_build` directory:

```bash
make
```

### Running with Docker

1.  **Build the Docker image:**

    ```bash
    make docker-build
    ```
2.  **Run the server with Docker:**

    ```bash
    make docker-run
    ```

    The server will be available at http://localhost:80.  Customize the port using the `PORT` variable:

    ```bash
    # The server will be available at http://localhost:3000
    make docker-run PORT=3000
    ```

## License

See the [LICENSE](LICENSE) file for details.