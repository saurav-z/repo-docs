# OSGameClones: Discover and Contribute to Open Source Game Clones

Explore a vast collection of open-source game clones and remakes, all in one place! This repository ([https://github.com/opengaming/osgameclones](https://github.com/opengaming/osgameclones)) serves as the source for [https://osgameclones.com](https://osgameclones.com), a comprehensive directory of games inspired by classic titles.

## Key Features

*   **Comprehensive Database:** Browse a curated list of open-source game clones and remakes.
*   **Community-Driven:**  Contribute to the database by adding new games or updating existing entries.
*   **YAML-Based Data:**  Game information is stored in easily accessible and modifiable YAML files.
*   **Validation:**  Data integrity is maintained through YAML schema validation for games and original titles.
*   **Easy Contribution:**  Add new games or references via pull requests or by submitting issues.

## Contributing to OSGameClones

Help build the ultimate resource for open-source gaming by contributing to the project!

### Adding a New Game Clone/Remake

1.  **Gather Information:** Collect relevant details about the game clone (title, developer, links, etc.).
2.  **Use the Form (Recommended):** Submit your entry using the [game form](https://osgameclones.com/add_game.html).
3.  **Direct File Editing (Advanced):**  Alternatively, directly edit the YAML files in the [`games/` directory](games/).  Your changes will be submitted as a pull request.
4.  **Validation:** All entries are validated against the rules in the [`schema/games.yaml`](schema/games.yaml) validation file.

### Adding a Reference to the Original Game

1.  **Gather Information:** Collect details about the original game (title, developer, etc.).
2.  **Use the Form:** Utilize the [add original form](https://osgameclones.com/add_original.html).
3.  **Direct File Editing (Advanced):** If the original game isn't already in the [`originals/` directory](originals/), create a new entry.
4.  **Validation:** All entries are validated against the rules in the [`schema/originals.yaml`](schema/originals.yaml) validation file.

### Pre-requisites

*   [poetry](https://python-poetry.org/)

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

### Building the Project

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

The server will be available at http://localhost:80. You can change the port using the **PORT** variable:

```bash
# The server will be available at http://localhost:3000
make docker-run PORT=3000
```

## License

See the [LICENSE](LICENSE) file for licensing information.