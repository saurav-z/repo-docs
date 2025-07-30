# OSGameClones: Discover and Contribute to Open Source Game Clones

This project is your one-stop resource for finding and contributing to open-source clones and remakes of classic video games.  [(Original Repository)](https://github.com/opengaming/osgameclones)

## Key Features:

*   **Comprehensive Database:** Explore a curated collection of open-source game clones, remakes, and reimplementations.
*   **Community Driven:** Contribute to the database by adding new games or improving existing information through pull requests and issues.
*   **YAML-Based Data:** Game and original game data are stored in easy-to-understand YAML files, making contributions straightforward.
*   **Validation & Structure:**  Data is validated against schema files (`schema/games.yaml` and `schema/originals.yaml`) to ensure data quality and consistency.
*   **Easy Contribution:**  Use pre-built forms to add games and original game references, or directly edit YAML files.

## Contributing

### Prerequisites

*   [Poetry](https://python-poetry.org/) (Python dependency management)

### Installation

1.  Clone the repository: `git clone <repository-url>`
2.  Navigate to the project directory: `cd osgameclones`
3.  Install dependencies: `poetry install`

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

2.  Run the server:

    ```bash
    make docker-run
    ```

    The server will be available at `http://localhost:80`. You can specify a different port:

    ```bash
    make docker-run PORT=3000 # Server available at http://localhost:3000
    ```

## Adding a New Game or Original Game Reference

*   **Add a Game Clone/Remake:**  Use the [game form](https://osgameclones.com/add_game.html) to submit information about a new game, or directly edit files in the `games/` directory.  Changes will be submitted as a pull request.
*   **Add an Original Game Reference:** Use the [add original form](https://osgameclones.com/add_original.html) to add a reference to the original game.  If an entry doesn't exist in the `originals/` directory, create a new one following the specified format.

##  Data Organization

*   **`games/`:** Contains YAML files describing the open-source game clones.
*   **`originals/`:** Contains YAML files describing the original games.
*   **`schema/games.yaml`:**  Defines the validation rules for game entries.
*   **`schema/originals.yaml`:** Defines the validation rules for original game entries.

## License

See the [LICENSE](LICENSE) file for details.