# Nix User Repository (NUR): Community-Driven Package Sharing

**NUR empowers Nix users to discover, share, and install community-contributed packages and configurations, expanding the Nix ecosystem.**

[Link to Original Repo: https://github.com/nix-community/NUR](https://github.com/nix-community/NUR)

## Key Features

*   **Community-Driven:** Access a vast library of user-contributed packages, configurations, and modules.
*   **Decentralized:** Share and install packages faster than through traditional channels, bypassing the Nixpkgs review process.
*   **Automatic Updates:**  NUR automatically checks repositories and performs evaluation checks to ensure stability.
*   **Flexible Installation:**  Integrate with Flakes, `packageOverrides`, NixOS configurations, and Home Manager.
*   **Easy Package Discovery:** Find packages through the NUR package search and the nur-combined repository.
*   **Repository Support:** Host NixOS modules, overlays, and library functions within your repository.
*   **Override Repositories:**  Test changes and customize your NUR experience by overriding existing repositories.

## Installation

### Using Flakes

Add NUR to your `flake.nix`:

```nix
{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    nur = {
      url = "github:nix-community/NUR";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };
};
```

Then, use either the overlay (`overlays.default`) or `legacyPackages.<system>`.

### Using `packageOverrides`

Add NUR to your `~/.config/nixpkgs/config.nix`:

```nix
{
  packageOverrides = pkgs: {
    nur = import (builtins.fetchTarball "https://github.com/nix-community/NUR/archive/main.tar.gz") {
      inherit pkgs;
    };
  };
}
```

For NixOS, add to `/etc/nixos/configuration.nix`:

```nix
{
  nixpkgs.config.packageOverrides = pkgs: {
    nur = import (builtins.fetchTarball "https://github.com/nix-community/NUR/archive/main.tar.gz") {
      inherit pkgs;
    };
  };
}
```

### Pinning

Pin versions for stable builds using `fetchTarball` with the `sha256` hash:

```nix
builtins.fetchTarball {
  url = "https://github.com/nix-community/NUR/archive/3a6a6f4da737da41e27922ce2cfacf68a109ebce.tar.gz";
  sha256 = "04387gzgl8y555b3lkz9aiw9xsldfg4zmzp930m62qw8zbrvrshd";
}
```

## How to Use

Install packages from the NUR namespace:

```console
$ nix-shell -p nur.repos.mic92.hello-nur
nix-shell> hello
Hello, NUR!
```

or

```console
$ nix-env -f '<nixpkgs>' -iA nur.repos.mic92.hello-nur
```

or

```nix
# configuration.nix
environment.systemPackages = with pkgs; [
  nur.repos.mic92.hello-nur
];
```

**Important Security Note:** *Always review packages before installation, as they are not reviewed by Nixpkgs.*

## Examples

*   **Devshell with a Single Package**

    ```nix
    {
      inputs = {
        nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
        flake-utils.url = "github:numtide/flake-utils";
        nur = {
          url = "github:nix-community/NUR";
          inputs.nixpkgs.follows = "nixpkgs";
        };
      };

      outputs = { self, nixpkgs, flake-utils, nur }:
        flake-utils.lib.eachDefaultSystem (system:
          let
            pkgs = import nixpkgs {
              inherit system;
              overlays = [ nur.overlay ];
            };
          in
          {
            devShells.default = pkgs.mkShell {
              packages = [ pkgs.nur.repos.mic92.hello-nur ];
            };
          }
        );
    }
    ```

*   **NixOS Configuration**

    ```nix
    {
      inputs = {
        nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
        flake-utils.url = "github:numtide/flake-utils";
        nur = {
          url = "github:nix-community/NUR";
          inputs.nixpkgs.follows = "nixpkgs";
        };
      };

      outputs = { self, nixpkgs, nur }: {
        nixosConfigurations.myConfig = nixpkgs.lib.nixosSystem {
          # ...
          modules = [
            # Adds the NUR overlay
            nur.modules.nixos.default
            # NUR modules to import
            nur.legacyPackages."${system}".repos.iopq.modules.xraya
            # This adds the NUR nixpkgs overlay.
            # Example:
            # ({ pkgs, ... }: {
            #   environment.systemPackages = [ pkgs.nur.repos.mic92.hello-nur ];
            # })
          ];
        };
      };
    };
    ```

*   **Home Manager Integration**

    ```nix
    let
      nur-no-pkgs = import (builtins.fetchTarball "https://github.com/nix-community/NUR/archive/main.tar.gz") {};
    in
    {
      imports = lib.attrValues nur-no-pkgs.repos.moredhel.hmModules.rawModules;

      services.unison = {
        enable = true;
        profiles = {
          org = {
            src = "/home/moredhel/org";
            dest = "/home/moredhel/org.backup";
            extraArgs = "-batch -watch -ui text -repeat 60 -fat";
          };
        };
      };
    }
    ```

## Finding Packages

*   [Packages search for NUR](https://nur.nix-community.org/)
*   [nur-combined](https://github.com/nix-community/nur-combined/search)

## Contributing Your Own Repository

1.  Create a Git repository with a `default.nix` file at the root.  Use the [repository template](https://github.com/nix-community/nur-packages-template) for guidance.

    *   **Do NOT** import from `<nixpkgs>`.  Use the `pkgs` argument.
    *   Each repository should return a set of Nix derivations.

    ```nix
    { pkgs }:
    {
      hello-nur = pkgs.callPackage ./hello-nur {};
    }
    ```

    Where `hello-nur` contains a package definition (example):

    ```nix
    { stdenv, fetchurl, lib }:

    stdenv.mkDerivation rec {
      # ... package definition ...
    }
    ```

2.  Add your repository to `repos.json`.

    ```console
    $ git clone --depth 1 https://github.com/nix-community/NUR
    $ cd NUR
    ```

    Edit `repos.json`:

    ```json
    {
        "repos": {
            "mic92": {
                "url": "https://github.com/Mic92/nur-packages"
            },
            "<fill-your-repo-name>": {
                "url": "https://github.com/<your-user>/<your-repo>"
            }
        }
    }
    ```

    Run:

    ```console
    $ ./bin/nur format-manifest # ensure repos.json is sorted alphabetically
    $ git add repos.json
    $ git commit -m "add <your-repo-name> repository"
    $ git push
    ```

3.  Open a pull request to [https://github.com/nix-community/NUR](https://github.com/nix-community/NUR).

## Advanced Repository Features

*   **Different root expressions:** Use the `file` option in `repos.json` to specify a non-`default.nix` file.
*   **Update NUR's lock file:**  Use the service at https://nur-update.nix-community.org/update after updating your repository.  See the [github page](https://github.com/nix-community/nur-update#nur-update-endpoint) for details.
*   **Troubleshooting Updates:** If packages aren't updating, check the [latest build job](https://github.com/nix-community/NUR/actions) for evaluation errors.
*   **Git Submodules:**  Set `submodules` to `true` in `repos.json`.
*   **NixOS Modules, Overlays, and Library Functions:** Organize these under `modules`, `overlays`, and `lib` attributes within your repository's `default.nix`.

## Overriding Repositories

Use `repoOverrides` within `packageOverrides` to test changes.

```nix
{
  packageOverrides = pkgs: {
    nur = import (builtins.fetchTarball "https://github.com/nix-community/NUR/archive/main.tar.gz") {
      inherit pkgs;
      repoOverrides = {
        mic92 = import ../nur-packages { inherit pkgs; };
        ## remote locations are also possible:
        # mic92 = import (builtins.fetchTarball "https://github.com/your-user/nur-packages/archive/main.tar.gz") { inherit pkgs; };
      };
    };
  };
}
```

## Contribution Guidelines

*   Ensure packages build and set `meta.broken = true` if broken.
*   Supply standard meta attributes for package discoverability.
*   Keep repositories lean.
*   Reuse packages from Nixpkgs when possible.

## Examples of Packages Suitable for NUR

*   Small-audience packages
*   Pre-releases
*   Legacy package versions
*   Automated package sets (e.g., from PyPI)
*   Opinionated patches
*   Experiments

## Contact

*   Matrix: [#nur:nixos.org](https://matrix.to/#/#nur:nixos.org)
*   Discourse: [https://discourse.nixos.org](https://discourse.nixos.org/)