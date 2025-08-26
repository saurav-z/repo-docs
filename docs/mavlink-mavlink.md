# MAVLink: The Open Standard for Drone Communication

MAVLink is a powerful, lightweight, and widely adopted message protocol, enabling seamless communication between drones and ground control stations.  [Learn more on GitHub](https://github.com/mavlink/mavlink).

[![Build Status](https://github.com/mavlink/mavlink/workflows/Test%20and%20deploy/badge.svg)](https://github.com/mavlink/mavlink/actions?query=branch%3Amaster)

## Key Features of MAVLink

*   **Lightweight & Efficient:** Optimized for resource-constrained systems with limited bandwidth, RAM, and flash memory.
*   **Header-Only Library:** Easy integration into various projects with minimal dependencies.
*   **Cross-Platform Support:** Generates code for multiple languages ([see supported languages](https://mavlink.io/en/#supported_languages)).
*   **XML-Based Message Definition:** Uses XML files to define message sets ("dialects"), promoting interoperability.
*   **Widely Adopted:** A field-proven standard used in numerous products, ensuring compatibility between manufacturers.
*   **Open Source and Community Driven:**  Benefit from a vibrant community and contribute to its ongoing development.

## Getting Started: Generate C Headers

Here's how to get started with the C library:

1.  **Install Dependencies:**

    ```bash
    sudo apt install python3-pip
    ```

2.  **Clone the Repository:**

    ```bash
    git clone https://github.com/mavlink/mavlink.git --recursive
    cd mavlink
    ```

3.  **Install Python Requirements:**

    ```bash
    python3 -m pip install -r pymavlink/requirements.txt
    ```

4.  **Generate C Headers:**

    ```bash
    python3 -m pymavlink.tools.mavgen --lang=C --wire-protocol=2.0 --output=generated/include/mavlink/v2.0 message_definitions/v1.0/common.xml
    ```

## Using with CMake

1.  **Install Headers Locally:**

    ```bash
    cmake -Bbuild -H. -DCMAKE_INSTALL_PREFIX=install -DMAVLINK_DIALECT=common -DMAVLINK_VERSION=2.0
    cmake --build build --target install
    ```

2.  **Use `find_package` in `CMakeLists.txt`:**

    ```cmake
    find_package(MAVLink REQUIRED)
    add_executable(my_program my_program.c)
    target_link_libraries(my_program PRIVATE MAVLink::mavlink)
    ```

3.  **Pass Install Directory to CMake:**

    ```bash
    cd ../my_program
    cmake -Bbuild -H. -DCMAKE_PREFIX_PATH=../mavlink/install
    ```

    *Note: MAVLink is header-only, so `target_link_libraries` doesn't link but makes the headers available.*

    For a full example, see the [examples/c](examples/c) directory.

## Further Resources

*   [Documentation/Website](https://mavlink.io/en/)
*   [Discussion/Support](https://mavlink.io/en/#support)
*   [Contributing](https://mavlink.io/en/contributing/contributing.html)
*   [License](https://mavlink.io/en/#license)
*   [Using C MAVLink Libraries (mavgen)](https://mavlink.io/en/mavgen_c/)
*   [Installing the MAVLink Toolchain](https://mavlink.io/en/getting_started/installation.html)
*   [Generating MAVLink Libraries](https://mavlink.io/en/getting_started/generate_libraries.html)
*   [Using MAVLink Libraries](https://mavlink.io/en/getting_started/use_libraries.html)