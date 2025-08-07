# OS Game Clones: Explore the World of Open-Source Game Remakes & Clones

Discover a comprehensive directory of open-source game clones and remakes, meticulously curated and ready for your exploration. ([Original Repository](https://github.com/opengaming/osgameclones))

## Key Features

*   **Extensive Database:** Browse a vast collection of open-source game clones, meticulously organized and categorized.
*   **Community-Driven:** Contribute to the project by adding new games, improving existing entries, and helping the community grow.
*   **YAML-Based Data:** All game and original game data are stored in easily accessible and editable YAML files for easy management and contribution.
*   **Validation:** Ensure data integrity with YAML schema validation for games and original games, guaranteeing data quality.
*   **Easy Contribution:** Submit new games and references to original games through pull requests or by using the provided forms.
*   **Docker Support:** Build and run the project easily with Docker for local development and deployment.

## How to Contribute

We encourage you to contribute to this project! Here's how:

### Adding a Game Clone/Remake

1.  **Use the Form:** Utilize the [game form](https://osgameclones.com/add_game.html) to submit information about a new game.
2.  **Directly Edit YAML:** For more advanced contributions, you can directly edit the YAML files in the [`games`](games/) directory.
3.  **Follow the Rules:** New games must adhere to the rules defined in the [`schema/games.yaml`](schema/games.yaml) validation file.

### Adding a Reference to an Original Game

1.  **Use the Form:** Fill out the [add original form](https://osgameclones.com/add_original.html) to add a reference to the original game.
2.  **Create New Entries (If Needed):** If the original game doesn't exist in the [`originals`](originals/) directory, create a new entry following the specified format.
3.  **Validate Your Entry:** Ensure your original game entry complies with the rules outlined in the [`schema/originals.yaml`](schema/originals.yaml) validation file.

## Development & Installation

### Prerequisites

*   [Poetry](https://python-poetry.org/)

### Installation

1.  Clone the repository:

    ```bash
    git clone <repository-url>  # Replace with the actual repository URL
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

### Running with Docker

1.  Build the Docker image:

    ```bash
    make docker-build
    ```

2.  Run the server with Docker:

    ```bash
    make docker-run
    ```

    The server will be accessible at http://localhost:80 (or the port you specify).

    Example with a custom port:

    ```bash
    make docker-run PORT=3000  # Server available at http://localhost:3000
    ```

## License

This project is licensed under the [LICENSE](LICENSE).