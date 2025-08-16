# MoviePilot: Automate Your Media Management (and More!)

[GitHub Repository](https://github.com/jxxghp/MoviePilot) | [Official Wiki](https://wiki.movie-pilot.org)

[![GitHub Repo stars](https://img.shields.io/github/stars/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot/network/members)
[![GitHub contributors](https://img.shields.io/github/contributors/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot/graphs/contributors)
[![GitHub repo size](https://img.shields.io/github/repo-size/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)
[![GitHub issues](https://img.shields.io/github/issues/jxxghp/MoviePilot?style=for-the-badge)](https://github.com/jxxghp/MoviePilot/issues)
[![Docker Pulls](https://img.shields.io/docker/pulls/jxxghp/moviepilot?style=for-the-badge)](https://hub.docker.com/repository/docker/jxxghp/moviepilot)
[![Docker Pulls V2](https://img.shields.io/docker/pulls/jxxghp/moviepilot-v2?style=for-the-badge)](https://hub.docker.com/repository/docker/jxxghp/moviepilot-v2)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20Synology-blue?style=for-the-badge)](https://github.com/jxxghp/MoviePilot)

MoviePilot is a powerful and user-friendly media management application designed to streamline your workflow with automation, a modern interface, and easy extensibility.  Built upon a foundation of core automation needs, MoviePilot offers a simplified and extensible architecture.

**Note:** This project is for educational and personal use only.  Please refrain from promoting this project on any domestic (Chinese) platforms.

**Stay Updated:** [MoviePilot Telegram Channel](https://t.me/moviepilot_channel)

## Key Features

*   **Modern Architecture:** Built with a clean, separation of concerns (front-end and back-end):
    *   **Frontend:** Uses Vue3 for a responsive and intuitive user interface.
    *   **Backend:** Built with FastAPI, offering robust API capabilities.
*   **Simplified Configuration:** Focuses on essential functionality with default values, making setup and usage straightforward.
*   **Enhanced User Interface:** A redesigned UI provides a more pleasant and efficient user experience.
*   **Extensible with Plugins:**  Develop custom plugins to extend MoviePilot's capabilities.

## Getting Started

*   **Official Documentation and Guides:**  Explore the comprehensive [MoviePilot Wiki](https://wiki.movie-pilot.org) for detailed installation and usage instructions.
*   **API Documentation:** Access the API documentation at [https://api.movie-pilot.org](https://api.movie-pilot.org) for more in-depth technical details.

### Prerequisites
*   Python 3.12
*   Node.js v20.12.1

### Installation and Setup

1.  **Clone the Main Project:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot
    ```
2.  **Clone the Resource Project:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot-Resources
    ```
    *   Copy the platform-specific `.so`, `.pyd`, or `.bin` files from the `MoviePilot-Resources/resources` directory to the `app/helper` directory within the main project.
3.  **Install Backend Dependencies:**
    ```bash
    cd MoviePilot/app # Navigate to the 'app' directory in your MoviePilot project
    pip install -r requirements.txt
    python3 main.py # Start the backend server (default port: 3001)
    # API Documentation: http://localhost:3001/docs
    ```
4.  **Clone and Run the Frontend:**
    ```bash
    git clone https://github.com/jxxghp/MoviePilot-Frontend
    cd MoviePilot-Frontend
    yarn
    yarn dev  # Start the frontend (accessible at http://localhost:5173)
    ```
5.  **Plugin Development:** Refer to the [Plugin Development Guide](https://wiki.movie-pilot.org/zh/plugindev) to create custom plugins in the `app/plugins` directory.

## Related Projects

*   [MoviePilot-Frontend](https://github.com/jxxghp/MoviePilot-Frontend)
*   [MoviePilot-Resources](https://github.com/jxxghp/MoviePilot-Resources)
*   [MoviePilot-Plugins](https://github.com/jxxghp/MoviePilot-Plugins)
*   [MoviePilot-Server](https://github.com/jxxghp/MoviePilot-Server)
*   [MoviePilot-Wiki](https://github.com/jxxghp/MoviePilot-Wiki)

## Disclaimer

*   This software is provided for educational and personal use only.
*   It is not intended for commercial use or any illegal activities.
*   Users are solely responsible for their use of the software.
*   The project is open-source.  Modifying and distributing the code to circumvent restrictions is discouraged.
*   The project does not accept donations or offer paid services.  Please be cautious of any misleading information.

## Contributors

<a href="https://github.com/jxxghp/MoviePilot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jxxghp/MoviePilot" />
</a>
```
Key improvements and SEO considerations:

*   **Strong Title & Hook:** The title includes relevant keywords ("MoviePilot," "Media Management," "Automate") and the hook sentence immediately grabs attention.
*   **Clear Headings:**  Uses consistent headings for better readability and SEO structure.
*   **Bulleted Key Features:**  Highlights the main selling points in an easily digestible format.
*   **Keyword Optimization:**  Uses relevant keywords throughout the description, like "automation," "user-friendly," "modern interface," and "extensibility."
*   **Links & Calls to Action:**  Includes clear links to the GitHub repository, Wiki, and API documentation.
*   **Concise Language:**  Avoids unnecessary jargon.
*   **Structure:**  Organized information for easier scanning.
*   **Contributor Section:** Keeps the existing contribution section.