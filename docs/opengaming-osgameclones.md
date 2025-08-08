# OSGameClones: Your Hub for Open Source Game Clones & Remakes

Explore a vast collection of open-source game clones and remakes, all in one convenient place! This project provides a comprehensive database and resources for finding and contributing to game projects that pay homage to classic titles. ([View the original repository](https://github.com/opengaming/osgameclones))

[![Build and Deploy](https://github.com/opengaming/osgameclones/actions/workflows/main.yml/badge.svg)](https://github.com/opengaming/osgameclones/actions/workflows/main.yml)

## Key Features

*   **Extensive Game Database:** Access a curated list of open-source game clones and remakes, meticulously organized.
*   **Easy Contribution:** Contribute new games, improve existing entries, and help grow the community.
*   **YAML-Based Data:**  All game and original game data are stored in easy-to-understand YAML files.
*   **Validation:**  Data is validated against defined schemas (`schema/games.yaml` and `schema/originals.yaml`) to ensure data integrity.
*   **Alphabetical Sorting:** Games are listed alphabetically for easy navigation, with the exception of ScummVM for better organization.
*   **Docker Support:** Easily run the project locally using Docker for development and testing.

## How to Contribute

We welcome contributions! Here's how you can get involved:

### Adding a Game Clone/Remake

1.  **Submit a new issue** using the [game form](https://osgameclones.com/add_game.html) provided.
2.  **Alternatively**, edit the YAML files directly in the [`games/` directory](games/) and submit a pull request.

### Adding a Reference to the Original Game

1.  Fill in the [add original form](https://osgameclones.com/add_original.html).
2.  If the original game doesn't exist, create a new entry following the provided format in the [`originals/` directory](originals/).

## Development & Setup

### Prerequisites
* [Poetry](https://python-poetry.org/)

### Installation

1.  Clone the repository:
    ```bash
    git clone [your repo url]
    cd osgameclones
    ```
2.  Install dependencies:
    ```bash
    poetry install
    ```

### Building the Project

Run the following command to build the project into the `_build` directory:

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
    *   The server will be available at http://localhost:80 by default.
    *   Customize the port using the `PORT` variable: `make docker-run PORT=3000` (server available at http://localhost:3000)

## License

This project is licensed under the [LICENSE](LICENSE)