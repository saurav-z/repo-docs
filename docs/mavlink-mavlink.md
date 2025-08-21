# MAVLink: Lightweight Communication for Drones and Ground Control Stations

**MAVLink is the leading open-source message protocol enabling seamless communication between drones and ground control stations, designed for efficiency and reliability.**

[View the original repository on GitHub](https://github.com/mavlink/mavlink)

MAVLink (Micro Air Vehicle Message Marshalling Library) is a header-only message library perfect for constrained bandwidth communication in the drone and robotics industries. It uses XML-defined message sets ("dialects") and Python tools to generate source code in multiple languages. This ensures interoperability between components from different manufacturers.

## Key Features of MAVLink:

*   **Lightweight & Efficient:** Designed for resource-constrained systems with limited RAM and flash memory.
*   **Cross-Platform:** Supports multiple programming languages, including C, Python, and more.
*   **XML-Based Message Definitions:** Utilizes XML files to define message structures, making it easy to manage and extend.
*   **Interoperability:** Enables communication between various drone components and ground control stations from different manufacturers.
*   **Field-Proven:** Widely used and deployed in numerous products within the drone and robotics industries.

## Getting Started with MAVLink (C Example)

Here's how to get started with MAVLink using C, for Ubuntu LTS 20.04 or 22.04:

1.  **Install Dependencies:**

    ```bash
    sudo apt install python3-pip
    ```

2.  **Clone the Repository:**

    ```bash
    git clone https://github.com/mavlink/mavlink.git --recursive
    cd mavlink
    ```

3.  **Install Python Dependencies:**

    ```bash
    python3 -m pip install -r pymavlink/requirements.txt
    ```

4.  **Generate C Headers:**
    Generate the C-library with the following command:
    ```bash
    python3 -m pymavlink.tools.mavgen --lang=C --wire-protocol=2.0 --output=generated/include/mavlink/v2.0 message_definitions/v1.0/common.xml
    ```

5.  **Using with CMake:**

    *   Install the headers locally, e.g., into the `install` directory:

        ```bash
        cmake -Bbuild -H. -DCMAKE_INSTALL_PREFIX=install -DMAVLINK_DIALECT=common -DMAVLINK_VERSION=2.0
        cmake --build build --target install
        ```

    *   In your `CMakeLists.txt`:

        ```cmake
        find_package(MAVLink REQUIRED)
        add_executable(my_program my_program.c)
        target_link_libraries(my_program PRIVATE MAVLink::mavlink)
        ```

    *   Pass the install directory to CMake:

        ```bash
        cd ../my_program
        cmake -Bbuild -H. -DCMAKE_PREFIX_PATH=../mavlink/install
        ```

    For more details, refer to the example files.

## Further Resources:

*   [Documentation/Website](https://mavlink.io/en/)
*   [Discussion/Support](https://mavlink.io/en/#support)
*   [Contributing](https://mavlink.io/en/contributing/contributing.html)
*   [License](https://mavlink.io/en/#license)
*   [Using C MAVLink Libraries (mavgen)](https://mavlink.io/en/mavgen_c/)
*   [Installing the MAVLink Toolchain](https://mavlink.io/en/getting_started/installation.html)
*   [Generating MAVLink Libraries](https://mavlink.io/en/getting_started/generate_libraries.html)
*   [Using MAVLink Libraries](https://mavlink.io/en/getting_started/use_libraries.html)