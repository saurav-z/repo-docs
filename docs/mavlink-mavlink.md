# MAVLink: The Lightweight Communication Protocol for Drones and Ground Control Stations

**MAVLink (Micro Air Vehicle Message Marshalling Library) is a powerful, open-source protocol enabling seamless communication between drones, ground control stations, and other components.** This versatile library is optimized for resource-constrained environments, making it ideal for a wide range of applications.

[View the original repository on GitHub](https://github.com/mavlink/mavlink)

## Key Features of MAVLink

*   **Lightweight & Efficient:** Designed for low-bandwidth communication, perfect for drones and embedded systems.
*   **Header-Only Library:** Simple to integrate, minimizing dependencies and compilation overhead.
*   **Message-Set Specifications:** Uses XML files to define message sets (dialects) for different systems, ensuring interoperability.
*   **Multi-Language Support:**  Generates code for various languages, including C, Python, and more. [See supported languages.](https://mavlink.io/en/#supported_languages)
*   **Field-Proven:** Widely adopted and used in numerous products, establishing a robust communication standard.
*   **Interoperability:** Facilitates communication between components from different manufacturers.
*   **Customizable:**  Easily adaptable to specific needs through custom message definitions.

## Getting Started with MAVLink (C Example)

Here's a quick guide to get you started generating C headers using the `common.xml` dialect on Ubuntu:

**Prerequisites:**

*   Ubuntu LTS 20.04 or 22.04
*   Python 3 and pip

**Installation:**

1.  **Install Dependencies:**

    ```bash
    sudo apt install python3-pip
    ```

2.  **Clone the Repository:**

    ```bash
    git clone https://github.com/mavlink/mavlink.git --recursive
    cd mavlink
    ```

3.  **Install Python Packages:**

    ```bash
    python3 -m pip install -r pymavlink/requirements.txt
    ```

4.  **Generate C Headers:**

    ```bash
    python3 -m pymavlink.tools.mavgen --lang=C --wire-protocol=2.0 --output=generated/include/mavlink/v2.0 message_definitions/v1.0/common.xml
    ```

**Using with CMake:**

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

3.  **Pass the Install Directory to CMake:**

    ```bash
    cd ../my_program
    cmake -Bbuild -H. -DCMAKE_PREFIX_PATH=../mavlink/install
    ```

   *For a complete working example, see [examples/c](examples/c).*

**Further Resources:**

*   **Using C MAVLink Libraries (mavgen):** [Using C MAVLink Libraries (mavgen)](https://mavlink.io/en/mavgen_c/)
*   **Installing the MAVLink Toolchain:** [Installing the MAVLink Toolchain](https://mavlink.io/en/getting_started/installation.html)
*   **Generating MAVLink Libraries:** [Generating MAVLink Libraries](https://mavlink.io/en/getting_started/generate_libraries.html)
*   **Using MAVLink Libraries:** [Using MAVLink Libraries](https://mavlink.io/en/getting_started/use_libraries.html)

## Key Links

*   **Documentation/Website:** [mavlink.io/en/](https://mavlink.io/en/)
*   **Discussion/Support:** [mavlink.io/en/#support](https://mavlink.io/en/#support)
*   **Contributing:** [mavlink.io/en/contributing/contributing.html](https://mavlink.io/en/contributing/contributing.html)
*   **License:** [mavlink.io/en/#license](https://mavlink.io/en/#license)