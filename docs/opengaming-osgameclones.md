# Discover the World of Open Source Game Clones: OSGameClones

**Explore a comprehensive database of open-source game clones and remakes, meticulously cataloged for enthusiasts and developers alike.** (Learn more at [https://osgameclones.com](https://osgameclones.com) - the source is here!)

[![Build and Deploy](https://github.com/opengaming/osgameclones/actions/workflows/main.yml/badge.svg)](https://github.com/opengaming/osgameclones/actions/workflows/main.yml)

This repository powers [https://osgameclones.com](https://osgameclones.com), a valuable resource for anyone interested in open-source game development and the preservation of classic gaming experiences. Contribute by adding new games, improving existing entries, or suggesting enhancements to this comprehensive collection.

## Key Features:

*   **Extensive Game Database:** A curated collection of open-source game clones and remakes, organized for easy browsing and discovery.
*   **Detailed Information:** Each entry includes references to the original games, ensuring proper attribution and context.
*   **Community Driven:**  Contribute directly by submitting pull requests or opening issues to enrich the database.
*   **YAML-Based Data:** Game and original game information is stored in YAML files, making data easily accessible and manageable.
*   **Validation:** All game and original game data is validated against schema rules for data integrity.

## How to Contribute

We welcome contributions!  Here's how you can help:

### Adding a Game Clone or Remake

1.  Fill out the [game form](https://osgameclones.com/add_game.html) to submit information about a new game.
2.  Alternatively, you can directly edit the files in the [`games/`](games/) directory. Your changes will be submitted as a pull request.

### Adding a Reference to an Original Game

1.  Use the [add original form](https://osgameclones.com/add_original.html) to add a reference to an original game.
2.  If there's no existing entry in the [`originals/`](originals/) directory, create a new entry using the following format.

## Development & Setup

### Prerequisites

*   [poetry](https://python-poetry.org/) -  A dependency management tool.

### Installation

1.  Clone this repository.
2.  Navigate into the repository directory in your terminal.
3.  Run:

    ```bash
    poetry install
    ```

### Building the Project

Build the project into the `_build` directory:

```bash
make
```

### Running the Server with Docker

1.  Build a Docker image:

    ```bash
    make docker-build
    ```

2.  Run the server using Docker:

    ```bash
    make docker-run
    ```

    The server will be available at `http://localhost:80`.  You can customize the port using the `PORT` variable:

    ```bash
    # The server will be available at http://localhost:3000
    make docker-run PORT=3000
    ```

## License

This project is licensed under the [LICENSE](LICENSE) file.

**[View the original repository on GitHub](https://github.com/opengaming/osgameclones)**