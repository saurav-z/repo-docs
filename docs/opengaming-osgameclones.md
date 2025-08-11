# OSGameClones: Explore and Contribute to Open Source Game Clones

Discover and contribute to a vast collection of open-source game clones and remakes, all in one place! Explore the OSGameClones project hosted on [GitHub](https://github.com/opengaming/osgameclones).

## Key Features

*   **Extensive Game Database:** Browse a curated collection of open-source game clones, remakes, and reimplementations.
*   **Easy Contribution:**  Add new games or improve existing entries via pull requests or issues.
*   **YAML-Based Data Storage:**  Game information and references to original games are stored in organized YAML files for easy access and modification.
*   **Validation and Schema:** Games and original game references are validated against defined schemas to ensure data integrity.
*   **Community-Driven:** Benefit from a community-driven effort to document and preserve classic games in open-source form.
*   **Docker Support:** Easily run the project with Docker for quick setup and testing.

## How to Contribute

### Add a Game Clone or Remake

1.  **Submit a New Issue:** Use the [game form](https://osgameclones.com/add_game.html) to submit details about a new game clone.
2.  **Directly Edit YAML Files:** For more experienced users, directly modify the YAML files in the `games/` directory. Your changes will be submitted as a pull request. All games are validated against the rules in the `schema/games.yaml` file.

### Add a Reference to the Original Game

1.  **Use the "Add Original" Form:**  Use the [add original form](https://osgameclones.com/add_original.html) to add references to the original games that clones are based on.
2.  **Create New Entry:** If the original game doesn't exist, create a new entry in the `originals/` directory following the specified format. All originals are validated against the rules in the `schema/originals.yaml` file.

### Prerequisites for Development

*   [poetry](https://python-poetry.org/)

### Installation

Clone the repository and run the following command inside the project directory:

```bash
poetry install
```

### Building

To build the project into the `_build` directory, run:

```bash
make
```

### Running with Docker

1.  **Build the Docker Image:**

    ```bash
    make docker-build
    ```

2.  **Run the Server with Docker:**

    ```bash
    make docker-run
    ```

    The server will be accessible at http://localhost:80.  You can change the port using the `PORT` variable.
    ```bash
    # The server will be available at http://localhost:3000
    make docker-run PORT=3000
    ```

## License

See the [LICENSE](LICENSE) file for details.