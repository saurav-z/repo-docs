# Discover Amazing Open-Source Game Clones & Remakes

Explore a curated database of open-source game clones and remakes, preserving classic gaming experiences and fostering community collaboration. This repository, the source behind [osgameclones.com](https://osgameclones.com), offers a comprehensive list of games and resources, perfect for gamers, developers, and anyone interested in game preservation.

[![Build and Deploy](https://github.com/opengaming/osgameclones/actions/workflows/main.yml/badge.svg)](https://github.com/opengaming/osgameclones/actions/workflows/main.yml)

## Key Features:

*   **Comprehensive Database:** Access a vast collection of game clones and remakes, meticulously documented.
*   **Community-Driven:** Contribute to the project by adding new games, updating information, and improving existing entries.
*   **YAML-Based Data:** Game and original game information are stored in easy-to-read YAML files, ensuring transparency and maintainability.
*   **Validation:** All game and original game entries are validated against schemas for data integrity.
*   **Easy Contribution:**  Submit new game entries using a provided form or by directly editing the YAML files and submitting a pull request.

## How to Contribute:

This project thrives on community contributions. Here's how you can get involved:

### Adding a Game Clone or Remake

1.  **Use the Game Form:** Utilize the [game form](https://osgameclones.com/add_game.html) to submit details about a new game.
2.  **Directly Edit YAML:**  For more advanced users, edit the YAML files directly within the [`games`](games/) directory and submit a pull request. Your changes will be validated.

### Adding a Reference to the Original Game

1.  **Use the Original Game Form:**  Submit information about the original game using the [add original form](https://osgameclones.com/add_original.html).
2.  **Create a New Entry:** If the original game doesn't exist in the [`originals`](originals/) directory, create a new entry following the provided format. All originals are validated.

## Setting up the Project:

### Prerequisites

*   [Poetry](https://python-poetry.org/)

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

### Running the Server with Docker

1.  Build the Docker image:

    ```bash
    make docker-build
    ```
2.  Run the server with Docker:

    ```bash
    make docker-run
    ```

    The server will be available at `http://localhost:80`.
3.  Customize the port:

    ```bash
    # The server will be available at http://localhost:3000
    make docker-run PORT=3000
    ```

## License

This project is licensed under the [LICENSE](LICENSE) (specify the actual license if it's known).

## Resources

*   **Games Directory:** [`games/`](games/)
*   **Originals Directory:** [`originals/`](originals/)
*   **Game Schema:** [`schema/games.yaml`](schema/games.yaml)
*   **Original Schema:** [`schema/originals.yaml`](schema/originals.yaml)
*   **Game Form:** [https://osgameclones.com/add_game.html](https://osgameclones.com/add_game.html)
*   **Original Form:** [https://osgameclones.com/add_original.html](https://osgameclones.com/add_original.html)

**[Back to the original repository](https://github.com/opengaming/osgameclones)**