# VRM Add-on for Blender

**Enhance your Blender workflow with the VRM Add-on, enabling seamless import, export, and editing of VRM models for virtual reality and beyond!** ([Original Repository](https://github.com/saturday06/VRM-Addon-for-Blender))

[![CI status](https://github.com/saturday06/VRM-Addon-for-Blender/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/saturday06/VRM-Addon-for-Blender/actions) [![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

This add-on empowers Blender users with comprehensive VRM functionality, allowing you to create, import, and export VRM models with ease.

## Key Features

*   **VRM Import and Export:** Seamlessly import and export VRM files directly within Blender.
*   **VRM Humanoid Support:**  Add and configure VRM Humanoid rigs for animation and rigging.
*   **MToon Shader Configuration:**  Create stunning anime-style materials using the integrated MToon shader support.
*   **Physics-Based Material Creation:** Generate realistic materials with physics-based rendering (PBR).
*   **Animation Support:** Animate your VRM models within Blender.
*   **Scripting API:** Automate tasks and extend functionality with Python scripting.

## Download

*   **Blender 4.2 or later:**  [🛠️**Blender Extensions Platform**](https://extensions.blender.org/add-ons/vrm)
*   **Blender 2.93 to 4.1:**  [🌐**The Official Site**](https://vrm-addon-for-blender.info)

## Tutorials

| Installation                                                                                                             | Create Simple VRM                                                                                                             | Create Humanoid VRM                                                                                                         |
| :-------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------: |
| <a href="https://vrm-addon-for-blender.info/en/installation?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/installation.gif"></a> | <a href="https://vrm-addon-for-blender.info/en/create-simple-vrm-from-scratch?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/simple.gif"></a> | <a href="https://vrm-addon-for-blender.info/en/create-humanoid-vrm-from-scratch?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/humanoid.gif"></a> |
| Create Physics-Based Material                                                                                                | Create Anime-Style Material                                                                                                        | Automation with Python Scripts                                                                                                    |
| <a href="https://vrm-addon-for-blender.info/en/material-pbr?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/material_pbr.gif"></a> | <a href="https://vrm-addon-for-blender.info/en/material-mtoon?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/material_mtoon.gif"></a> | <a href="https://vrm-addon-for-blender.info/en/scripting-api?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/scripting_api.gif"></a> |
| VRM Animation                                                                                                             | Development How-To                                                                                                             |                                                                                                                                  |
| <a href="https://vrm-addon-for-blender.info/en/animation?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/animation.gif"></a>  | <a href="https://vrm-addon-for-blender.info/en/development?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/animation.gif"></a>  |                                                                                                                                  |

## Development

The add-on's core code resides in the [`src/io_scene_vrm`](https://github.com/saturday06/VRM-Addon-for-Blender/tree/main/src/io_scene_vrm) directory. For efficient testing and development, create a symbolic link to this directory within Blender's add-ons folder.

Detailed instructions for setting up your development environment can be found in the [development environment setup documentation](https://vrm-addon-for-blender.info/en/development?locale_redirection).

### Creating a Development Link

Instructions provided for Linux, macOS, and Windows PowerShell to create symbolic links for Blender versions 4.2+ and earlier.

---

**Note:** This README is available in both English and Japanese.