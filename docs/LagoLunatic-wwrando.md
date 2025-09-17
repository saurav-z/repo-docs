# Conquer the Seas in a Fresh Adventure with the Wind Waker Randomizer!

Dive into a unique and unpredictable journey through the Great Sea with the Wind Waker Randomizer, a tool that reshapes the beloved Zelda classic.  Explore a randomized world, discover new item locations, and experience the game like never before!  For more information and to get started, visit the original repository: [https://github.com/LagoLunatic/wwrando](https://github.com/LagoLunatic/wwrando).

## Key Features:

*   **Randomized Item Placement:** Every playthrough is unique, with items appearing in unexpected locations, keeping you on your toes.
*   **Open World from the Start:** Experience the freedom of exploration with an open world from the very beginning of your adventure.
*   **Enhanced Gameplay:** Enjoy increased sailing and text speeds for a more streamlined experience and fewer cutscenes.
*   **Customizable Difficulty:** Tailor the randomization to your preference, limiting progress items to specific location types like dungeons or secret caves.
*   **Guaranteed Completion:** The randomizer ensures every seed is completable, eliminating the need for glitches.

## Getting Started

*   **Compatibility:** The Wind Waker Randomizer is designed for the North American GameCube version of The Legend of Zelda: The Wind Waker (MD5: d8e4d45af2032a081a0f446384e9261b). Other versions are not supported.
*   **Download:** Get the latest version of the randomizer here: [https://github.com/LagoLunatic/wwrando/releases/latest](https://github.com/LagoLunatic/wwrando/releases/latest)
*   **Official Website:** For detailed instructions, FAQs, and more, visit the official website: [https://lagolunatic.github.io/wwrando/](https://lagolunatic.github.io/wwrando/)

## Troubleshooting

*   **Antivirus Issues:** If the randomizer fails to launch or is flagged as malware, add an exception in your antivirus software.
*   **Stuck in a Seed?:** Consult the FAQ page on the official website first. If you believe you've found a bug, report it, including the seed's permalink, on the issues page.

## Community

*   **Discord:** Connect with other players and get your questions answered in the official Wind Waker Randomizer Discord server: [https://discord.gg/r2963mt](https://discord.gg/r2963mt)

## Credits

The Wind Waker Randomizer is a community effort, thanks to the following contributors:

*   LagoLunatic (Creator & Programmer)
*   Aelire (additional programming)
*   CryZe (event flag documentation)
*   EthanArmbrust (Mac and Linux support)
*   Fig (additional programming)
*   Gamma / SageOfMirrors (custom model conversion, file format documentation, additional programming)
*   Hypatia (textures)
*   JarheadHME (additional programming)
*   LordNed (file format documentation)
*   MelonSpeedruns (game design suggestions, graphic design)
*   nbouteme (starting items programming)
*   tanjo3 (CTMC and hint programming)
*   TrogWW (additional programming)
*   wooferzfg (additional programming)

## Running from Source (Advanced Users)

Follow these steps to run the randomizer from source code:

1.  **Install Git:** Download and install Git from [https://git-scm.com/downloads](https://git-scm.com/downloads).
2.  **Clone the Repository:** Open a command prompt and run `git clone --recurse-submodules https://github.com/LagoLunatic/wwrando.git`.
3.  **Install Python:** Download and install Python 3.12 from [https://www.python.org/downloads/release/python-3121/](https://www.python.org/downloads/release/python-3121/) or use your OS's package manager.
4.  **Install Dependencies:** Open the `wwrando` folder in a command prompt and run the appropriate command:
    *   **Windows:** `py -3.12 -m pip install -r requirements.txt`
    *   **Mac:** `python3 -m pip install -r requirements.txt`
    *   **Linux:** `python3 -m pip install -r requirements.txt --user`
5.  **Run the Randomizer:**
    *   **Windows:** `py -3.12 wwrando.py`
    *   **Mac/Linux:** `python3 wwrando.py`
6.  **(Optional) Install Full Dependencies:** For faster texture recoloring and building a distributable version, install the packages in `requirements_full.txt` using the same process as `requirements.txt`.
7.  **(Windows 8 or below)** If you are on Windows 8 or below, use `requirements_qt5.txt`/`requirements_qt5_full.txt` instead of the normal requirements files.