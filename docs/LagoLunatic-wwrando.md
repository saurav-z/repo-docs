# Wind Waker Randomizer: Experience The Legend of Zelda Anew!

**Embark on a thrilling, unpredictable adventure as the Wind Waker Randomizer shuffles item locations, creating a fresh and engaging experience with every playthrough.**

[View the original repository on GitHub](https://github.com/LagoLunatic/wwrando)

## Key Features

*   **Randomized Item Locations:** Discover a completely new game with randomized item placements, ensuring a unique experience every time.
*   **Open-World Exploration:** Enjoy the freedom of open-world exploration from the very beginning of the game.
*   **Streamlined Gameplay:** Experience faster sailing and text speeds, along with the option to remove cutscenes for a more immersive experience.
*   **Customizable Randomization:** Control the scope of randomization by limiting progress items to specific location types (dungeons, caves, etc.)
*   **Guaranteed Completable Seeds:** The randomizer ensures every playthrough is beatable, without requiring glitches.

## Getting Started

To begin your randomized Wind Waker adventure, visit the official website for detailed instructions and downloads: [https://lagolunatic.github.io/wwrando/](https://lagolunatic.github.io/wwrando/)

You can also download the latest version of the randomizer directly: [https://github.com/LagoLunatic/wwrando/releases/latest](https://github.com/LagoLunatic/wwrando/releases/latest)

**Important Notes:**

*   **Supported Version:** The randomizer is designed for the North American GameCube version of The Legend of Zelda: The Wind Waker (MD5: d8e4d45af2032a081a0f446384e9261b).  European, Japanese, and HD versions are not compatible.
*   **Antivirus Issues:** If the randomizer fails to launch, your antivirus software may be incorrectly flagging it. Add an exception for the randomizer to resolve this.

## Troubleshooting & Support

*   **Stuck in a Seed?** Consult the FAQ on the official website for assistance: [https://lagolunatic.github.io/wwrando/faq/](https://lagolunatic.github.io/wwrando/faq/)
*   **Encountering a Bug?** Report any bugs you find on the issue tracker, including the seed permalink: [https://github.com/LagoLunatic/wwrando/issues](https://github.com/LagoLunatic/wwrando/issues)
*   **Community:** Join the Wind Waker Randomizer Discord server for questions, gameplay, and races: [https://discord.gg/r2963mt](https://discord.gg/r2963mt)

## Credits

This project is brought to you by a dedicated team:

*   **LagoLunatic** (Creator & Programmer)
*   **Aelire** (Additional Programming)
*   **CryZe** (Event Flag Documentation)
*   **EthanArmbrust** (Mac and Linux Support)
*   **Fig** (Additional Programming)
*   **Gamma / SageOfMirrors** (Custom Model Conversion, File Format Documentation, Additional Programming)
*   **Hypatia** (Textures)
*   **JarheadHME** (Additional Programming)
*   **LordNed** (File Format Documentation)
*   **MelonSpeedruns** (Game Design Suggestions, Graphic Design)
*   **nbouteme** (Starting Items Programming)
*   **tanjo3** (CTMC and Hint Programming)
*   **TrogWW** (Additional Programming)
*   **wooferzfg** (Additional Programming)

## Running From Source (Advanced Users)

If you're interested in contributing or experimenting with the latest development version, follow these instructions:

1.  **Install Git:** [https://git-scm.com/downloads](https://git-scm.com/downloads)
2.  **Clone the Repository:** `git clone --recurse-submodules https://github.com/LagoLunatic/wwrando.git`
3.  **Install Python 3.12:** [https://www.python.org/downloads/release/python-3121/](https://www.python.org/downloads/release/python-3121/) (or use your Linux distribution's package manager, like `sudo apt-get install python3.12`)
4.  **Navigate to the `wwrando` folder in a command prompt.**
5.  **Install Dependencies:**
    *   **Windows:** `py -3.12 -m pip install -r requirements.txt`
    *   **macOS:** `python3 -m pip install -r requirements.txt`
    *   **Linux:** `python3 -m pip install -r requirements.txt --user`
    *   **Windows 8 or below:** Use `requirements_qt5.txt` instead of `requirements.txt`
6.  **Run the Randomizer:**
    *   **Windows:** `py -3.12 wwrando.py`
    *   **macOS/Linux:** `python3 wwrando.py`
7.  **Optional: Install Full Requirements:** Install `requirements_full.txt` for faster texture recoloring and build support (same installation process as `requirements.txt`).