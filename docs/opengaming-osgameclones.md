# Explore Open Source Game Clones: Relive Classic Games with Modern Tech

[![Build and Deploy](https://github.com/opengaming/osgameclones/actions/workflows/main.yml/badge.svg)](https://github.com/opengaming/osgameclones/actions/workflows/main.yml)

This repository, the engine behind [osgameclones.com](https://osgameclones.com), offers a comprehensive database of open-source game clones, allowing you to rediscover beloved classic games built with modern technology. Contribute to the project by submitting pull requests or opening issues!

## Key Features of the Open Source Game Clones Database

*   **Extensive Database:** Discover a curated collection of open-source remakes and clones of classic video games.
*   **Community-Driven:** Contribute to the project by adding new games or updating information about existing ones.
*   **YAML-Based Data:** Game and original game information are stored in easy-to-understand YAML files within the [`games`][games] and [`originals`][originals] directories.
*   **Data Validation:** All game and original game entries are validated against schema files ( [`schema/games.yaml`][schema_games] and [`schema/originals.yaml`][schema_originals]) ensuring data integrity.
*   **Alphabetical Sorting:** Games are generally sorted alphabetically for easy navigation, with the exception of ScummVM entries.

## How to Contribute: Help Build the Ultimate Game Clone Resource

We welcome contributions! Here's how you can get involved:

### Adding a New Game Clone or Remake

1.  **Submit a New Game Entry:** Use the [game form][game_form] to submit information about a new game clone.
2.  **Direct File Editing:** For more advanced users, you can directly edit the YAML files in the [`games`][games] directory and submit a pull request.

### Adding a Reference to the Original Game

1.  **Submit a New Original Game Entry:** Utilize the [add original form][original_form] to add information about the original game.
2.  **Create a New Entry:** If the original game doesn't exist, create a new entry following the format outlined in the `originals` directory.

## Technical Information and Setup

### Prerequisites

*   [poetry][poetry]

### Installation

1.  Clone the repository.
2.  Navigate into the project directory.
3.  Run `poetry install`.

### Building the Project

To build the project into the `_build` directory, simply run:

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

    The server will be accessible at `http://localhost:80`. You can customize the port using the `PORT` variable:

    ```bash
    # Access the server at http://localhost:3000
    make docker-run PORT=3000
    ```

## License

This project is licensed under the [LICENSE][license].

[games]: games/
[originals]: originals/
[schema_games]: schema/games.yaml
[schema_originals]: schema/originals.yaml
[game_form]: https://osgameclones.com/add_game.html
[original_form]: https://osgameclones.com/add_original.html
[license]: LICENSE

[python]: https://www.python.org
[poetry]: https://python-poetry.org/