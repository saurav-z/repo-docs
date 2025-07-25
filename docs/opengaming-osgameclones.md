# OSGameClones: Your Comprehensive Directory of Open Source Game Clones and Remakes

Discover and explore a vast collection of open-source game clones and remakes at [osgameclones.com](https://osgameclones.com), a community-driven resource for classic gaming experiences.  This repository fuels the website, offering a curated database for developers, gamers, and enthusiasts.

[![Build and Deploy](https://github.com/opengaming/osgameclones/actions/workflows/main.yml/badge.svg)](https://github.com/opengaming/osgameclones/actions/workflows/main.yml)

## Key Features

*   **Extensive Game Database:** Explore a growing directory of game clones and remakes, categorized and referenced by their original game counterparts.
*   **Community-Driven:** Contribute to the project by adding new games, improving information, and helping to build a valuable resource for the open-source gaming community.
*   **Easy Contribution:** Contribute by creating an issue using the [game form](https://osgameclones.com/add_game.html) or the [original form](https://osgameclones.com/add_original.html). Even better, submit a pull request directly!
*   **YAML-Based Data:** Game and original game data are stored in easy-to-read YAML files, making it simple to understand and contribute to the database.  Files are located in the [`games`](games/) and [`originals`](originals/) directories.
*   **Data Validation:**  All game entries and original game references are validated against schemas to ensure data integrity (see [`schema/games.yaml`](schema/games.yaml) and [`schema/originals.yaml`](schema/originals.yaml)).
*   **Docker Support:** Easily run the website locally using Docker for development and testing.

## Contributing

We welcome contributions from the community!  Whether you want to add a new game, improve existing information, or report an issue, your contributions are valuable.

### How to Contribute

1.  **Add a Game Clone or Remake:**  Use the [game form](https://osgameclones.com/add_game.html) to submit information, or directly edit files within the [`games`](games/) directory.
2.  **Add a Reference to an Original Game:** Use the [add original form](https://osgameclones.com/add_original.html) to reference the original game that a clone or remake is based on.  If the original game isn't already listed, create a new entry in the [`originals`](originals/) directory, following the existing format.

### Development Prerequisites

*   [poetry](https://python-poetry.org/) (for dependency management)

### Development Setup and Building

1.  **Install Dependencies:** Clone the repository and run `poetry install` in the project directory.
2.  **Build the Project:**  Run `make` to build the project and generate the website in the `_build` directory.

### Running the Server with Docker

1.  **Build the Docker Image:**  Run `make docker-build`.
2.  **Run the Server:** Run `make docker-run`. The server will be available at http://localhost:80 by default.  Customize the port using the **PORT** variable:  `make docker-run PORT=3000` (server available at http://localhost:3000)

## License

This project is licensed under the [LICENSE](LICENSE).

**[View the Source Code on GitHub](https://github.com/opengaming/osgameclones)**