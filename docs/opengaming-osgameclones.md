# Discover & Explore Open Source Game Clones: A Comprehensive Database

**Looking for free and open-source alternatives to your favorite classic games?** This project provides a curated, searchable database of game clones and remakes, empowering you to rediscover gaming classics in a community-driven, open-source environment. Visit the live site at [https://osgameclones.com](https://osgameclones.com/)!

[![Build and Deploy](https://github.com/opengaming/osgameclones/actions/workflows/main.yml/badge.svg)](https://github.com/opengaming/osgameclones/actions/workflows/main.yml)

This repository serves as the source code and data repository for [osgameclones.com](https://osgameclones.com), a comprehensive directory of open-source game clones.  Contribute to the project by adding new games, improving existing information, or suggesting new features.

## Key Features of the Open Source Game Clones Database:

*   **Extensive Game Listings:** Browse a vast collection of open-source clones and remakes, covering a wide range of genres and classic titles.
*   **Organized Data:**  Game information is stored in easily accessible and well-structured YAML files for easy understanding and contribution.
*   **Community-Driven:**  Actively contribute to the database by submitting pull requests and creating issues.
*   **Validation & Quality:**  Game entries are validated against defined schema to ensure data accuracy and consistency.
*   **Searchable & User-Friendly:** Provides easy access to find clones and remakes of your favorite games.

## How to Contribute

This project thrives on community contributions.  Here's how you can get involved:

### Adding a Game Clone or Remake

1.  **Create a New Issue:** Use the [game form](https://osgameclones.com/add_game.html) to provide details about the game clone.
2.  **Direct Editing:**  Alternatively, edit the YAML files in the [`games`](games/) directory directly and submit a pull request.

### Adding a Reference to an Original Game

1.  **Use the Original Game Form:** Fill out the [add original form](https://osgameclones.com/add_original.html) to add details of the original game.
2.  **Create a New Entry:** If the original game is not already listed, create a new entry in the [`originals`](originals/) directory.

## Technical Details

### Prerequisites

*   [poetry](https://python-poetry.org/)

### Installation

1.  Clone the repository.
2.  Navigate to the project directory and run:

    ```bash
    poetry install
    ```

### Building

Build the project to the `_build` directory:

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

    The server will be accessible at `http://localhost:80`.  Customize the port using the `PORT` variable:

    ```bash
    make docker-run PORT=3000
    # Server will be available at http://localhost:3000
    ```

## License

This project is licensed under the [LICENSE](LICENSE) (MIT License).

**[Visit the original repository on GitHub](https://github.com/opengaming/osgameclones) for source code and more information.**