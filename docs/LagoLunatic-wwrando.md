# The Legend of Zelda: Wind Waker Randomizer - Unleash a Fresh Adventure!

Dive into a thrilling new take on a classic with the **Wind Waker Randomizer**, offering randomized item placement and a host of exciting enhancements to your gameplay. ([Original Repo](https://github.com/LagoLunatic/wwrando))

## Key Features

*   **Randomized Item Placement:** Experience a unique playthrough every time, with items shuffled across the entire game world.
*   **Open World from the Start:** Explore the Great Sea freely from the beginning of your adventure.
*   **Streamlined Gameplay:** Enjoy faster sailing and text speeds, alongside the removal of many cutscenes.
*   **Guaranteed Completable:** Every randomized seed is designed to be beatable without relying on glitches.
*   **Customizable Logic:** Tailor your experience by limiting the locations where progress items can appear, creating a balanced challenge.

## Important Information

*   **Compatibility:** This randomizer is compatible with the **North American GameCube version** of *The Legend of Zelda: The Wind Waker* (MD5: d8e4d45af2032a081a0f446384e9261b). European, Japanese versions, and Wind Waker HD are not supported.
*   **Antivirus:** If the randomizer fails to launch, your antivirus software may be falsely detecting it as a threat. Create an exception for the randomizer to resolve this.
*   **Troubleshooting:** Consult the [FAQ](https://lagolunatic.github.io/wwrando/faq/) for common issues. If you encounter a bug, report it with the seed's permalink on the [issues page](https://github.com/LagoLunatic/wwrando/issues).
*   **Download:** Get the latest version of the randomizer [here](https://github.com/LagoLunatic/wwrando/releases/latest).

## Community & Support

*   **Discord:** Join the official [Discord server](https://discord.gg/r2963mt) to connect with other players, get help, and organize play sessions.

## Credits

The Wind Waker Randomizer is a collaborative effort, thanks to the dedication of:

*   LagoLunatic (Creator & Programmer)
*   And many more contributors: Aelire, CryZe, EthanArmbrust, Fig, Gamma / SageOfMirrors, Hypatia, JarheadHME, LordNed, MelonSpeedruns, nbouteme, tanjo3, TrogWW, and wooferzfg.

## Running from Source (For Advanced Users)

Follow these steps to run the randomizer from its source code:

1.  **Install Git:** Download and install Git from [git-scm.com/downloads](https://git-scm.com/downloads).
2.  **Clone the Repository:** Open a command prompt and run: `git clone --recurse-submodules https://github.com/LagoLunatic/wwrando.git`
3.  **Install Python 3.12:** Download and install Python 3.12 from [python.org/downloads/release/python-3121/](https://www.python.org/downloads/release/python-3121/). If you're on Linux, run `sudo apt-get install python3.12`.
4.  **Install Dependencies:** Navigate to the `wwrando` folder in your command prompt and run:
    *   Windows: `py -3.12 -m pip install -r requirements.txt`
    *   Mac: `python3 -m pip install -r requirements.txt`
    *   Linux: `python3 -m pip install -r requirements.txt --user`
5.  **Run the Randomizer:**
    *   Windows: `py -3.12 wwrando.py`
    *   Mac/Linux: `python3 wwrando.py`
6.  **Optional:** Install `requirements_full.txt` for additional features. If you're on Windows 8 or below, use `requirements_qt5.txt`/`requirements_qt5_full.txt`.