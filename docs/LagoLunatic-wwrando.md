# Wind Waker Randomizer: Re-Experience a Classic in a Brand New Way!

[View the original repository](https://github.com/LagoLunatic/wwrando)

This is the ultimate tool for any Legend of Zelda: Wind Waker fan looking to revitalize their gameplay, randomizing item locations, opening the world, and speeding up the experience! 

## Key Features:

*   **Completely Randomized Item Placement:** Every playthrough offers a fresh and unpredictable experience.
*   **Open World from the Start:** Explore the Great Sea without restrictions.
*   **Faster Sailing & Text Speed:** Spend less time waiting and more time adventuring.
*   **Customizable Logic:** Tailor the item pool to the locations you want to check for items.
*   **Guaranteed Completable Seeds:** Enjoy a frustration-free experience.
*   **North American GameCube Version Compatible:** Supports the original game.

## Getting Started

1.  **Download the Randomizer:** [https://github.com/LagoLunatic/wwrando/releases/latest](https://github.com/LagoLunatic/wwrando/releases/latest)
2.  **Official Website for Details:** [https://lagolunatic.github.io/wwrando/](https://lagolunatic.github.io/wwrando/)
3.  **Supported Game Version:** North American GameCube version only (MD5: d8e4d45af2032a081a0f446384e9261b)

## Troubleshooting

### Randomizer Won't Launch?

Your antivirus software may be incorrectly flagging the randomizer as malware. Add an exception/exclusion for the randomizer in your antivirus software.

### Stuck in a Seed?

*   **Check the FAQ:** Consult the Frequently Asked Questions page on the official website: [https://lagolunatic.github.io/wwrando/faq/](https://lagolunatic.github.io/wwrando/faq/)
*   **Report Bugs:** If you believe you've found a bug, report it here: [https://github.com/LagoLunatic/wwrando/issues](https://github.com/LagoLunatic/wwrando/issues) Include the permalink for the seed.

## Community & Support

*   **Join the Discord Server:** Get help, find players, and discuss the game: [https://discord.gg/r2963mt](https://discord.gg/r2963mt)

## Credits

A huge thank you to the following contributors for making this project a reality:

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

For those who want to contribute or experiment, here's how to run the development version from source. *Requires Python 3.12 and Git.*

1.  **Clone the Repository:**
    ```bash
    git clone --recurse-submodules https://github.com/LagoLunatic/wwrando.git
    ```
2.  **Install Python 3.12:** [https://www.python.org/downloads/release/python-3121/](https://www.python.org/downloads/release/python-3121/)
    *   Linux users can use: `sudo apt-get install python3.12`
3.  **Navigate to the wwrando folder** in your command prompt.
4.  **Install Dependencies:**
    *   **Windows:** `py -3.12 -m pip install -r requirements.txt`
    *   **Mac:** `python3 -m pip install -r requirements.txt`
    *   **Linux:** `python3 -m pip install -r requirements.txt --user`
    *   **Windows 8 or below:** Use `requirements_qt5.txt`
5.  **Run the Randomizer:**
    *   **Windows:** `py -3.12 wwrando.py`
    *   **Mac/Linux:** `python3 wwrando.py`
6.  **Optional (For Enhanced Features):** Install `requirements_full.txt` using the same process.