# NUR: The Nix User Repository

**Expand your Nix package collection with the community-driven Nix User Repository (NUR) and access a vast library of user-contributed packages.**

[View the original repository](https://github.com/nix-community/NUR)

NUR is a community-driven meta-repository for Nix packages, offering access to user-created package descriptions (Nix expressions). Unlike Nixpkgs, packages in NUR are built from source and are not reviewed by Nixpkgs maintainers, enabling faster sharing and wider availability of software.

**Key Features:**

*   **Community-Driven:** Access packages contributed by the Nix community.
*   **Decentralized Package Sharing:** Easily share and install packages through user repositories.
*   **Automatic Updates:** NUR regularly checks and evaluates repositories for updates.
*   **Flexible Installation:** Install via flakes, `packageOverrides`, or direct use in your NixOS configuration.
*   **Wide Package Selection:** Find packages that may not be in Nixpkgs, including pre-releases, older versions, or specialized software.
*   **Support for NixOS Modules, Overlays, and Library Functions:** Organize more than just packages for easy integration.

## Installation

### Using Flakes

Include NUR in your `flake.nix`:

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

Then, either the overlay (`overlays.default`) or `legacyPackages.<system>` can be used.

### Using `packageOverrides`

Add the following to your `~/.config/nixpkgs/config.nix`:

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

### Pinning for Stability

Pinning with `sha256` ensures build reproducibility:

```nix
builtins.fetchTarball {
  url = "https://github.com/nix-community/NUR/archive/3a6a6f4da737da41e27922ce2cfacf68a109ebce.tar.gz";
  sha256 = "04387gzgl8y555b3lkz9aiw9xsldfg4zmzp930m62qw8zbrvrshd";
}
```

## How to Use

Install and use packages from the NUR namespace:

```bash
$ nix-shell -p nur.repos.mic92.hello-nur
nix-shell> hello
Hello, NUR!
```

```bash
$ nix-env -f '<nixpkgs>' -iA nur.repos.mic92.hello-nur
```

```nix
# configuration.nix
environment.systemPackages = with pkgs; [
  nur.repos.mic92.hello-nur
];
```

**Important:** *Always review package expressions before installing.*

## Examples

### Using a Single Package in a Devshell (Flake)

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

### Using in NixOS

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
      modules = [
        nur.modules.nixos.default
        nur.legacyPackages."${system}".repos.iopq.modules.xraya
      ];
    };
  };
}
```

### Integrating with Home Manager

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
*   [nur-combined repository](https://github.com/nix-community/nur-combined/search)

## Adding Your Own Repository

1.  Create a repository with a `default.nix` file.  Consider using the [repository template](https://github.com/nix-community/nur-packages-template).
2.  **Important:** Import dependencies from `pkgs` (passed as an argument).  Avoid `with import <nixpkgs> {};`.
3.  Your `default.nix` should return a set of Nix derivations.

    ```nix
    { pkgs }:
    {
      hello-nur = pkgs.callPackage ./hello-nur {};
    }
    ```

4.  In your `hello-nur/default.nix`, define your package:

    ```nix
    { stdenv, fetchurl, lib }:

    stdenv.mkDerivation rec {
      pname = "hello";
      version = "2.10";
      # ... (package definition)
    }
    ```
5.  Add your repository to `repos.json`:

    ```bash
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
            "<your-repo-name>": {
                "url": "https://github.com/<your-user>/<your-repo>"
            }
        }
    }
    ```

6.  Run:
    ```bash
    ./bin/nur format-manifest
    git add repos.json
    git commit -m "add <your-repo-name> repository"
    git push
    ```

7.  Open a pull request to [https://github.com/nix-community/NUR](https://github.com/nix-community/NUR).

### Repository Options

*   **`file`:** Use a different root expression file.
*   **`submodules`: `true`:** Enable Git submodule support.

### Updating Lock File

To update NUR faster after your changes, use the service at: [https://nur-update.nix-community.org/](https://nur-update.nix-community.org/)
(More details at: [https://github.com/nix-community/nur-update#nur-update-endpoint](https://github.com/nix-community/nur-update#nur-update-endpoint))

### Troubleshooting

**Why are my NUR packages not updating?**

*   Check the [latest build job](https://github.com/nix-community/NUR/actions) for evaluation errors.
*   Ensure your package metadata uses the correct Nixpkgs attributes.
*   Avoid `builtins.fetch*` functions. Use `pkgs.fetch*` instead.

#### Local Evaluation Check

Test the evaluation within your repository using the [check evaluation](https://github.com/nix-community/nur-packages-template/blob/main/.github/workflows/build.yml) task.

### Overriding Repositories

Override repositories using the `repoOverrides` argument.
This is useful for testing changes before publishing.

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

### Overriding Repositories with Flakes (Experimental)

Override repositories using `repoOverrides` in flakes.

```nix
{
  inputs.nur.url = "github:nix-community/NUR";
  inputs.paul.url = "path:/some_path/nur-paul"; # example: a local nur.repos.paul for development

  outputs = {self, nixpkgs, nur, paul }: {

  system = "x86_64-linux";

  nurpkgs = import nixpkgs { inherit system; };

  ...
  modules = [
       {
         nixpkgs.config.packageOverrides = pkgs: {
            nur = import nur {
              inherit pkgs nurpkgs;
              repoOverrides = { paul = import paul { inherit pkgs; }; };
            };
          };
        }
  ];
  ...
}
```

The repository must contain a `flake.nix` file in addition to a `default.nix`:  [flake.nix example](https://github.com/Mic92/nur-packages/blob/master/flake.nix)

## Contribution Guidelines

*   Ensure packages build and set `meta.broken = true` if not.
*   Use standard [Nixpkgs meta attributes](https://nixos.org/nixpkgs/manual/#sec-standard-meta-attributes).
*   Keep repositories slim.
*   Reuse packages from Nixpkgs when possible.

## Contact

Join us on [#nur:nixos.org](https://matrix.to/#/#nur:nixos.org) or [https://discourse.nixos.org](https://discourse.nixos.org/).