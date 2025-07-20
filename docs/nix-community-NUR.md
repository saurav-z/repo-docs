# Nix User Repository (NUR): Extend Your Nix Ecosystem

Nix User Repository (NUR) is a community-driven platform for sharing and installing Nix packages not found in Nixpkgs, providing faster access to user-created software. [Visit the original repository](https://github.com/nix-community/NUR).

**Key Features:**

*   **Community-Driven:** Access packages created and maintained by the Nix community.
*   **Decentralized Package Sharing:** Share and install packages quickly, bypassing the formal review process of Nixpkgs.
*   **Automatic Updates:** NUR automatically checks for updates in user repositories and performs evaluation checks to ensure package integrity.
*   **Flake and Package Override Support:** Seamlessly integrate NUR into your Nix configurations using flakes or package overrides.
*   **Flexible Installation:** Install packages via `nix-shell`, `nix-env`, or direct integration with NixOS configuration.
*   **NixOS Modules, Overlays, and Library Function Support:** Easily include NixOS modules, overlays, and library functions from user repositories.
*   **Package Discovery:** Find packages through the [NUR package search](https://nur.nix-community.org/) or the `nur-combined` repository.

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

Then, use `overlays.default` or `legacyPackages.<system>`.

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

Pinning is recommended for stable builds.

```nix
builtins.fetchTarball {
  url = "https://github.com/nix-community/NUR/archive/3a6a6f4da737da41e27922ce2cfacf68a109ebce.tar.gz";
  sha256 = "04387gzgl8y555b3lkz9aiw9xsldfg4zmzp930m62qw8zbrvrshd";
}
```

## How to Use

Install or use packages from the `nur` namespace:

```bash
$ nix-shell -p nur.repos.mic92.hello-nur
nix-shell> hello
Hello, NUR!
```

or

```bash
$ nix-env -f '<nixpkgs>' -iA nur.repos.mic92.hello-nur
```

or in your `configuration.nix`:

```nix
environment.systemPackages = with pkgs; [
  nur.repos.mic92.hello-nur
];
```

**Important:** Check expressions before installing as NUR doesn't regularly scan for malicious content.

## Examples

*   **Using a single package in a devshell:**

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

*   **Using a Flake in NixOS:**

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

*   **Integrating with Home Manager:**

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

## Adding Your Repository

1.  Create a Git repository with a `default.nix` file. Use the [repository template](https://github.com/nix-community/nur-packages-template).
2.  Avoid `with import <nixpkgs> {};`. Instead, use the `pkgs` argument.
3.  Each repository should return a set of Nix derivations:

```nix
{ pkgs }:
{
  hello-nur = pkgs.callPackage ./hello-nur {};
}
```

4.  Add your repository details to `repos.json` in NUR:

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

5.  Run `./bin/nur format-manifest` and commit the changes to `repos.json` (NOT `repos.json.lock`).
6.  Open a pull request to [https://github.com/nix-community/NUR](https://github.com/nix-community/NUR).

### Using a different nix file as root expression

Set the `file` option:

```json
{
    "repos": {
        "mic92": {
            "url": "https://github.com/Mic92/nur-packages",
            "file": "subdirectory/default.nix"
        }
    }
}
```

### Updating the Lock File

Use [https://nur-update.nix-community.org/](https://nur-update.nix-community.org/) to update NUR faster:

```bash
curl -XPOST https://nur-update.nix-community.org/update?repo=mic92
```

### Troubleshooting Updates

Errors during evaluation will prevent updates. Common issues:

*   Wrong license attributes.
*   Using `builtins.fetch*` instead of `pkgs.fetch*`.

Check the [latest build job](https://github.com/nix-community/NUR/actions) to debug.  You can locally [check evaluation](https://github.com/nix-community/nur-packages-template/blob/main/.github/workflows/build.yml) with `nix-env -f . -qa \* --meta ...`.

### Git Submodules

Enable submodules with `submodules`:

```json
{
    "repos": {
        "mic92": {
            "url": "https://github.com/Mic92/nur-packages",
            "submodules": true
        }
    }
}
```

### NixOS Modules, Overlays, and Library Functions

*   **NixOS Modules:** Place in the `modules` attribute:

```nix
{ pkgs }: {
  modules = import ./modules;
}
```

*   **Overlays:** Use the `overlays` attribute:

```nix
# default.nix
{
  overlays = {
    hello-overlay = import ./hello-overlay;
  };
}
```

*   **Library Functions:** Put reusable nix functions in the `lib` attribute:

```nix
{ pkgs }:
with pkgs.lib;
{
  lib = {
    hexint = x: hexvals.${toLower x};

    hexvals = listToAttrs (imap (i: c: { name = c; value = i - 1; })
      (stringToCharacters "0123456789abcdef"));
  };
}
```

## Overriding Repositories

### Using `packageOverrides`

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

### Overriding with Flakes (Experimental)

*   Using `packageOverrides`:

```nix
{
  inputs.nur.url = "github:nix-community/NUR";
  inputs.paul.url = "path:/some_path/nur-paul";

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

*   With overlay:

```nix
{
  modules = [
    {
      nixpkgs.overlays = [
        (final: prev: {
          nur = import nur {
            nurpkgs = prev;
            pkgs = prev;
            repoOverrides = { paul = import paul { pkgs = prev; }; };
          };
        })
      ];
    }
    ...
  ];
}
```

Repositories must contain `flake.nix`.

## Contribution Guidelines

*   Ensure packages build and set `meta.broken = true` if not.
*   Use Nixpkgs [meta attributes](https://nixos.org/nixpkgs/manual/#sec-standard-meta-attributes).
*   Keep repositories lean.
*   Reuse Nixpkgs packages where possible.

Examples of suitable packages:

*   Packages for small audiences
*   Pre-releases
*   Older versions
*   Generated package sets (e.g., from PyPI or CPAN)
*   Software with custom patches
*   Experiments

## Contact

*   Matrix: [#nur:nixos.org](https://matrix.to/#/#nur:nixos.org)
*   Discourse: [https://discourse.nixos.org/](https://discourse.nixos.org/)