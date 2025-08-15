# MoviePilot: Automate Your Movie & TV Show Management

MoviePilot is a powerful, open-source solution designed to streamline your movie and TV show management, built for ease of use and extensibility.  For more details, visit the original repository: [MoviePilot on GitHub](https://github.com/jxxghp/MoviePilot).

[![GitHub Repo stars](https://img.shields.io/github/stars/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub forks](https://img.shields.io/github/forks/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub contributors](https://img.shields.io/github/contributors/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub repo size](https://img.shields.io/github/repo-size/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub issues](https://img.shields.io/github/issues/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot/issues)
[![Docker Pulls](https://img.shields.io/docker/pulls/jxxghp/moviepilot?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot)
[![Docker Pulls V2](https://img.shields.io/docker/pulls/jxxghp/moviepilot-v2?style=for-the-badge)](https://hub.docker.com/r/jxxghp/moviepilot-v2)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20Synology-blue?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)

MoviePilot builds upon the foundation of [NAStool](https://github.com/NAStool/nas-tools), focusing on core automation needs while simplifying and improving maintainability.

**Disclaimer: This project is intended for learning and educational purposes only.  Please refrain from promoting or using this project on any platform within China.**

## Key Features

*   **Modern Architecture:** Leverages a front-end and back-end separation using FastAPI and Vue3 for a responsive and efficient user experience.
*   **Simplified Configuration:** Focuses on essential features, minimizing complex settings and offering sensible defaults for easy setup.
*   **Enhanced User Interface:** Features a redesigned, intuitive, and visually appealing user interface.
*   **Extensible via Plugins:**  Offers a plugin architecture for customized functionality.

## Installation and Usage

Detailed instructions and guidance can be found in the official wiki: [MoviePilot Wiki](https://wiki.movie-pilot.org)

### Prerequisites

*   Python 3.12
*   Node.js v20.12.1

### Setup Instructions

1.  **Clone the main project:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot
    ```

2.  **Clone the resources project:** (containing platform-specific libraries) and copy the relevant files.
    ```bash
    git clone https://github.com/jxxghp/MoviePilot-Resources
    ```
    Copy the `.so`, `.pyd`, or `.bin` files from the `MoviePilot-Resources/resources` directory to the `app/helper` directory within the main project, matching your platform.

3.  **Install backend dependencies and run the server:**
    ```bash
    cd MoviePilot  # Navigate into your project directory
    pip install -r requirements.txt
    python3 main.py
    ```
    The backend server will run on port 3001 by default. API documentation is available at: `http://localhost:3001/docs`

4.  **Clone and Run the Frontend**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot-Frontend
    cd MoviePilot-Frontend
    yarn
    yarn dev
    ```
    The frontend can be accessed at: `http://localhost:5173`

5.  **Developing Plugins:**
    Consult the [Plugin Development Guide](https://wiki.movie-pilot.org/zh/plugindev) for instructions on developing plugins and place your plugin code in the `app/plugins` directory.

## Related Projects

*   [MoviePilot-Frontend](https://github.com/jxxghp/MoviePilot-Frontend)
*   [MoviePilot-Resources](https://github.com/jxxghp/MoviePilot-Resources)
*   [MoviePilot-Plugins](https://github.com/jxxghp/MoviePilot-Plugins)
*   [MoviePilot-Server](https://github.com/jxxghp/MoviePilot-Server)
*   [MoviePilot-Wiki](https://github.com/jxxghp/MoviePilot-Wiki)

## Disclaimer

*   This software is solely for learning and educational purposes.  Any commercial use or participation in illegal activities is strictly prohibited. The software developers and maintainers are not responsible for user actions.
*   The source code is open-source.  Users who modify and redistribute the software are solely responsible for any resulting issues. Avoid bypassing or modifying user authentication mechanisms.
*   This project does not accept donations and does not offer any paid services. Please exercise caution and avoid any misleading solicitations.

## Contributors

<a href="https://github.com/jxxghp/MoviePilot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jxxghp/MoviePilot" />
</a>
```
Key improvements and SEO considerations:

*   **Clear & Concise Title:**  "MoviePilot: Automate Your Movie & TV Show Management" - includes primary keywords.
*   **One-Sentence Hook:** Provides an immediate understanding of the project's purpose.
*   **Descriptive Headings:**  Using appropriate H2 and H3 headings for structure and SEO.
*   **Bulleted Key Features:** Highlights benefits, improving readability and SEO keyword usage.
*   **Keyword Integration:**  "Automation," "movie," "TV shows," and other relevant terms are incorporated.
*   **Links to Core Resources:** Links to the GitHub repo and the wiki are clearly highlighted.
*   **Simplified Installation:** The installation steps have been made more explicit and user-friendly.
*   **Removed Unnecessary Sections:**  Cleaned up the unnecessary blank lines and redundant phrasing.
*   **Clearer Disclaimer:**  The disclaimer is improved, emphasizing the project's intended use.
*   **SEO-friendly badges:** Moved the badges up to near the title and project description.