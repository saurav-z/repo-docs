# VRM Add-on for Blender

> Bring your 3D creations to life with the VRM Add-on, empowering you to import, export, and manipulate VRM models directly within Blender!

[![CI status](https://github.com/saturday06/VRM-Addon-for-Blender/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/saturday06/VRM-Addon-for-Blender/actions) [![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

Easily add VRM functionality to Blender with this versatile add-on! Create, import, export, and modify VRM models seamlessly within Blender's intuitive interface.

**[Visit the GitHub Repository](https://github.com/saturday06/VRM-Addon-for-Blender)**

## Key Features

*   **Import and Export VRM:** Easily import and export VRM models, enabling seamless integration with the VRM ecosystem.
*   **VRM Humanoid Support:** Add and configure VRM Humanoid rigs within Blender, simplifying character animation.
*   **MToon Shader Integration:** Apply and customize MToon shaders for a stylized anime look, perfect for virtual characters.
*   **PBR Material Support:** Create physics-based rendering materials for realistic appearances.
*   **Scripting API:** Automate tasks and extend functionality using Python scripts.
*   **VRM Animation:** Animate your VRM models directly within Blender.

## Download

Get the add-on for your Blender version:

*   **Blender 4.2 or later:**  Download via the [🛠️**Blender Extensions Platform**](https://extensions.blender.org/add-ons/vrm).
*   **Blender 2.93 to 4.1:** Download via the [🌐**Official Site**](https://vrm-addon-for-blender.info).

## Tutorials

Explore a variety of tutorials to get you started:

| Installation                                                                                                        | Create Simple VRM                                                                                                          | Create Humanoid VRM                                                                                                             |
| :-------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------: |
| <a href="https://vrm-addon-for-blender.info/en/installation?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/installation.gif"></a> | <a href="https://vrm-addon-for-blender.info/en/create-simple-vrm-from-scratch?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/simple.gif"></a> | <a href="https://vrm-addon-for-blender.info/en/create-humanoid-vrm-from-scratch?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/humanoid.gif"></a> |
| Create Physics Based Material                                                                                       | Create Anime Style Material                                                                                                 | Automation with Python Scripts                                                                                                   |
| <a href="https://vrm-addon-for-blender.info/en/material-pbr?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/material_pbr.gif"></a> | <a href="https://vrm-addon-for-blender.info/en/material-mtoon?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/material_mtoon.gif"></a> | <a href="https://vrm-addon-for-blender.info/en/scripting-api?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/scripting_api.gif"></a> |
| VRM Animation                                                                                                        | Development How-To                                                                                                            |                                                                                                                                   |
| <a href="https://vrm-addon-for-blender.info/en/animation?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/animation.gif"></a> | <a href="https://vrm-addon-for-blender.info/en/development?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/animation.gif"></a> |                                                                                                                                   |

## Development

The source code is located in the `main` branch of the repository.  For efficient development, create a symbolic link to the `src/io_scene_vrm` folder in your Blender addons directory.

For more advanced development, use [astral.sh/uv](https://docs.astral.sh/uv/). Refer to the [tutorial](https://vrm-addon-for-blender.info/en/development?locale_redirection) for more details.

```text
git checkout main

# Blender 4.2 or later

# Linux
ln -Ts "$PWD/src/io_scene_vrm" "$HOME/.config/blender/BLENDER_VERSION/extensions/user_default/vrm"
# macOS
ln -s "$PWD/src/io_scene_vrm" "$HOME/Library/Application Support/Blender/BLENDER_VERSION/extensions/user_default/vrm"
# Windows PowerShell
New-Item -ItemType Junction -Path "$Env:APPDATA\Blender Foundation\Blender\BLENDER_VERSION\extensions\user_default\vrm" -Value "$(Get-Location)\src\io_scene_vrm"
# Windows Command Prompt
mklink /j "%APPDATA%\Blender Foundation\Blender\BLENDER_VERSION\extensions\user_default\vrm" src\io_scene_vrm

# Blender 4.1.1 or earlier

# Linux
ln -Ts "$PWD/src/io_scene_vrm" "$HOME/.config/blender/BLENDER_VERSION/scripts/addons/io_scene_vrm"
# macOS
ln -s "$PWD/src/io_scene_vrm" "$HOME/Library/Application Support/Blender/BLENDER_VERSION/scripts/addons/io_scene_vrm"
# Windows PowerShell
New-Item -ItemType Junction -Path "$Env:APPDATA\Blender Foundation\Blender\BLENDER_VERSION\scripts\addons\io_scene_vrm" -Value "$(Get-Location)\src\io_scene_vrm"
# Windows Command Prompt
mklink /j "%APPDATA%\Blender Foundation\Blender\BLENDER_VERSION\scripts\addons\io_scene_vrm" src\io_scene_vrm
```

---