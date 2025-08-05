# Open Source Game Clones: Discover and Contribute to Classic Game Remakes

[![Build and Deploy](https://github.com/opengaming/osgameclones/actions/workflows/main.yml/badge.svg)](https://github.com/opengaming/osgameclones/actions/workflows/main.yml)

This repository powers [osgameclones.com](https://osgameclones.com), a comprehensive database of open-source game clones and remakes.  Contribute to preserving gaming history by adding or improving information about these fascinating projects.

## Key Features

*   **Extensive Database:** Explore a curated collection of open-source clones and remakes of classic games.
*   **Easy Contribution:**  Contribute by submitting pull requests or opening issues to add new games or update existing information.
*   **YAML-Based Data:**  Game and original game data are stored in well-organized YAML files within the `games/` and `originals/` directories, making it easy to understand and modify the data.
*   **Validation:**  Game and original game entries are validated against schemas (`schema/games.yaml` and `schema/originals.yaml`) to ensure data integrity.
*   **Interactive Forms:** Use the [game form](https://osgameclones.com/add_game.html) and [add original form](https://osgameclones.com/add_original.html) for easy data submission.
*   **Docker Support:** Easily run the project locally with Docker for development and testing.

## Games Database Structure

The core of the project is built around YAML files:

*   **`games/`**: Contains detailed information about each game clone, including links, descriptions, and more.  Data is alphabetized for easy browsing, except for the large ScummVM entry.
*   **`originals/`**:  Provides references to the original games that the clones are based on.

## Contributing to the Project

We welcome contributions!  Here's how you can help:

### Adding a Game Clone/Remake

1.  Fill out the [game form](https://osgameclones.com/add_game.html) to submit details about a new clone.
2.  Alternatively, directly edit the YAML files in the `games/` directory and submit a pull request. Your changes will be validated against the defined schema.

### Adding a Reference to the Original Game

1.  Fill out the [add original form](https://osgameclones.com/add_original.html) to provide information about the original game that a clone is based on.
2.  If the original game isn't already listed, create a new entry in the `originals/` directory following the specified format.  All entries are validated against a schema.

### Prerequisites

*   [poetry](https://python-poetry.org/)

### Installation

1.  Clone this repository.
2.  Navigate to the project directory in your terminal.
3.  Run `poetry install`.

### Building the Project

Build the project into the `_build` directory using:

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
    The server will be available at http://localhost:80.

3.  Customize the port:

    ```bash
    make docker-run PORT=3000
    ```
    (The server will then be available at http://localhost:3000)

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file.

**[Back to the Repository](https://github.com/opengaming/osgameclones)**