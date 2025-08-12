# Open Source Game Clones: Recreating Gaming Classics 🕹️

**Explore and contribute to a curated database of open-source game clones and remakes, bringing beloved classic games to new platforms.**  This project, hosted on GitHub, provides a comprehensive listing of games, allowing users to find, contribute, and learn about open-source game development.  Check out the live site at [https://osgameclones.com](https://osgameclones.com).

[![Build and Deploy](https://github.com/opengaming/osgameclones/actions/workflows/main.yml/badge.svg)](https://github.com/opengaming/osgameclones/actions/workflows/main.yml)

## Key Features

*   **Extensive Game Database:**  Browse a growing collection of open-source game clones and remakes, referencing their original game counterparts.
*   **YAML-Based Data:** Game and original game information is stored in easily accessible YAML files (`games/` and `originals/`), making it easy to contribute.
*   **Contribution Encouraged:**  Submit new games, update existing entries, and improve the information through pull requests or by opening issues.
*   **Validation:**  All game entries are validated against defined schemas (`schema/games.yaml` and `schema/originals.yaml`) ensuring data integrity.
*   **Easy Setup:**  Simple instructions for local setup, including Docker support for easy deployment.

## Contributing to the Project

Want to help expand the database or improve the information?  Here's how you can contribute:

### Adding a New Game Clone

1.  **Use the Game Form:**  Fill out the [game form](https://osgameclones.com/add_game.html) to submit details about a new clone.
2.  **Directly Edit YAML:**  Alternatively, directly modify the YAML files within the `games/` directory.  Your changes will be submitted as a pull request.

### Adding References to Original Games

1.  **Use the Original Game Form:**  Fill out the [add original form](https://osgameclones.com/add_original.html) to add a reference to the original game.
2.  **Create a New Entry:** If there's no existing entry for the original game in the `originals/` directory, you can create a new one by following the specified format.

## Development and Setup

### Prerequisites

*   [poetry](https://python-poetry.org/)

### Installation

Clone the repository and run `poetry install` inside the project directory.

```bash
git clone <repository-url>  # Replace with the actual repository URL
cd osgameclones
poetry install
```

### Building the Project

To build the project into the `_build` directory:

```bash
make
```

### Running the Server with Docker

1.  **Build the Docker Image:**

    ```bash
    make docker-build
    ```

2.  **Run the Server:**

    ```bash
    make docker-run
    ```

    The server will be available at `http://localhost:80`.  You can customize the port using the `PORT` variable:

    ```bash
    # The server will be available at http://localhost:3000
    make docker-run PORT=3000
    ```

## License

See [LICENSE][license]

## Project Resources

*   [Games Directory][games]
*   [Originals Directory][originals]
*   [Game Schema (games.yaml)][schema_games]
*   [Original Schema (originals.yaml)][schema_originals]
*   [Original Project Repository](https://github.com/opengaming/osgameclones)