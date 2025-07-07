# NUR: The Nix User Repository

**Expand your Nix package library with community-driven packages and experimental software using NUR, a decentralized meta-repository.** ([Back to Original Repo](https://github.com/nix-community/NUR))

Nix User Repository (NUR) is a community-driven meta-repository for Nix packages, offering a wider selection of software. It provides access to user repositories containing package descriptions (Nix expressions), allowing users to install packages not found in Nixpkgs, often with faster updates.

## Key Features

*   **Community-Driven:** Share and discover packages maintained by the Nix community.
*   **Decentralized:** Install packages from various user repositories.
*   **Faster Updates:** Access new packages and pre-releases more quickly than through Nixpkgs.
*   **Flexible:** Support for packages, NixOS modules, overlays, and library functions.
*   **Flake and PackageOverrides Support:** Easily integrate NUR into your Nix configurations.
*   **Automated Checks:**  Regularly checks repositories for updates and performs evaluation checks to ensure packages are buildable.

## Installation

### Using Flakes

Integrate NUR with your `flake.nix`:

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

Use the overlay (`overlays.default`) or `legacyPackages.<system>`.

### Using `packageOverrides`

Add this to `~/.config/nixpkgs/config.nix` to make NUR accessible:

```nix
{
  packageOverrides = pkgs: {
    nur = import (builtins.fetchTarball "https://github.com/nix-community/NUR/archive/main.tar.gz") {
      inherit pkgs;
    };
  };
}
```

For NixOS, add the following to your `/etc/nixos/configuration.nix`:

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

Pin the NUR version for reliability:

```nix
builtins.fetchTarball {
  url = "https://github.com/nix-community/NUR/archive/3a6a6f4da737da41e27922ce2cfacf68a109ebce.tar.gz";
  sha256 = "04387gzgl8y555b3lkz9aiw9xsldfg4zmzp930m62qw8zbrvrshd";
}
```

## How to Use

Install packages using `nix-shell`, `nix-env`, or NixOS configuration.

```console
$ nix-shell -p nur.repos.mic92.hello-nur
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

**Important Note:** NUR does not regularly check for malicious content; review expressions before installation.

### Using a single package in a devshell

Here's how to add a package from NUR to a devshell using flakes.

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

### Using the flake in NixOS

Integrate NUR overlays and modules:

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

Add modules to `imports` to configure services:

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
*   [nur-combined](https://github.com/nix-community/nur-combined/search) (GitHub search)

## Contributing Your Own Repository

1.  Create a repository with a `default.nix`.
2.  Use the [repository template](https://github.com/nix-community/nur-packages-template).
3.  Import dependencies from Nixpkgs using the `pkgs` argument.
4.  Structure each repository to return a set of Nix derivations (packages).

```nix
{ pkgs }:
{
  hello-nur = pkgs.callPackage ./hello-nur {};
}
```

Add your repository to `repos.json`:

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
        "<your-repo-name>": {
            "url": "https://github.com/<your-user>/<your-repo>"
        }
    }
}
```

Run:

```console
$ ./bin/nur format-manifest
$ git add repos.json
$ git commit -m "add <your-repo-name> repository"
$ git push
```

Open a pull request to [https://github.com/nix-community/NUR](https://github.com/nix-community/NUR).

### Advanced Options

*   **`file`**: Use a different root expression file.
*   **`submodules`**: Enable Git submodule fetching.

### Overriding Repositories

Use `repoOverrides` to test changes before publishing:

```nix
{
  packageOverrides = pkgs: {
    nur = import (builtins.fetchTarball "https://github.com/nix-community/NUR/archive/main.tar.gz") {
      inherit pkgs;
      repoOverrides = {
        mic92 = import ../nur-packages { inherit pkgs; };
      };
    };
  };
}
```

Overriding repositories with Flakes: (experimental) Use the `repoOverrides` argument.

## Contribution Guidelines

*   Ensure packages build and set `meta.broken` if not.
*   Use standard Nixpkgs meta attributes.
*   Keep repositories slim.
*   Reuse Nixpkgs packages when possible.

## Examples of Packages Suitable for NUR

*   Packages for a smaller audience
*   Pre-releases
*   Legacy software
*   Automatically generated package sets
*   Software with opinionated patches
*   Experiments

## Contact

*   Matrix: [#nur:nixos.org](https://matrix.to/#/#nur:nixos.org)
*   Discourse: [https://discourse.nixos.org/](https://discourse.nixos.org/)