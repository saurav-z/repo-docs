# NUR: The Nix User Repository

**Expand your Nix package options with NUR, a community-driven repository for user-contributed packages.**

NUR (Nix User Repository) empowers the Nix community by providing a platform to share and access user-created Nix packages. Unlike Nixpkgs, NUR offers a faster, more decentralized way to discover and install packages built from source.  Find the original repository [here](https://github.com/nix-community/NUR).

**Key Features:**

*   **Community-Driven:** Access a wide range of packages contributed by the Nix community.
*   **Decentralized:**  Easily discover and install packages not found in Nixpkgs.
*   **Up-to-date:** NUR automatically checks and evaluates repositories for updates.
*   **Flexible Installation:** Integrates seamlessly with flakes, packageOverrides, and Home Manager.
*   **Package Discovery:** Find packages via the dedicated [Packages search for NUR](https://nur.nix-community.org/) or through [nur-combined](https://github.com/nix-community/nur-combined/search)

## Installation

### Using Flakes

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

Use either the overlay (`overlays.default`) or `legacyPackages.<system>`.

### Using `packageOverrides`

To make NUR accessible, add the following to `~/.config/nixpkgs/config.nix`:

```nix
{
  packageOverrides = pkgs: {
    nur = import (builtins.fetchTarball "https://github.com/nix-community/NUR/archive/main.tar.gz") {
      inherit pkgs;
    };
  };
}
```

For NixOS, add the following to `/etc/nixos/configuration.nix`:

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

Pinning is recommended to avoid cache issues. Example:

```nix
builtins.fetchTarball {
  url = "https://github.com/nix-community/NUR/archive/3a6a6f4da737da41e27922ce2cfacf68a109ebce.tar.gz";
  sha256 = "04387gzgl8y555b3lkz9aiw9xsldfg4zmzp930m62qw8zbrvrshd";
}
```

## How to Use

Install packages from the NUR namespace.

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

```console
# configuration.nix
environment.systemPackages = with pkgs; [
  nur.repos.mic92.hello-nur
];
```

***Important Security Note:** Always review package expressions before installing them. NUR does not provide regular security checks.*

### Using a Single Package in a Devshell

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

### Using the Flake in NixOS

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
*   [nur-combined](https://github.com/nix-community/nur-combined/search)

## Contributing Your Own Repository

1.  Create a repository with a `default.nix`.  Use the [repository template](https://github.com/nix-community/nur-packages-template).
2.  Avoid importing packages with `with import <nixpkgs> {};`. Use the `pkgs` argument instead. Each repo should return a set of Nix derivations.
    ```nix
    { pkgs }:
    {
      hello-nur = pkgs.callPackage ./hello-nur {};
    }
    ```
3.  Add your repository to `repos.json`:

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
4.  Run `./bin/nur format-manifest` and commit the changes to `repos.json` (but not `repos.json.lock`).
5.  Open a pull request on [https://github.com/nix-community/NUR](https://github.com/nix-community/NUR).

*Repositories should be buildable on Nixpkgs unstable.*

### Using a Different Nix File

Set the `file` option in `repos.json`.

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

### Updating NUR's Lock File

Use the service at https://nur-update.nix-community.org/update?repo=<your_repo_name> to update faster.  Check the [github page](https://github.com/nix-community/nur-update#nur-update-endpoint) for further details.

### Why are my NUR Packages Not Updating?

*   Ensure the evaluation of your repository does not contain any errors.
*   Common errors include using a wrong license or using `builtins.fetch` instead of `pkgs.fetch`.
*   Check the [latest build job](https://github.com/nix-community/NUR/actions).

#### Local Evaluation Check

```sh
nix-env -f . -qa \* --meta \
  --allowed-uris https://static.rust-lang.org \
  --option restrict-eval true \
  --option allow-import-from-derivation true \
  --drv-path --show-trace \
  -I nixpkgs=$(nix-instantiate --find-file nixpkgs) \
  -I ./ \
  --json | jq -r 'values | .[].name'
```

### Git Submodules

Set `submodules`:

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

### NixOS Modules, Overlays and Library Functions

*   Use the `modules`, `overlays`, and `lib` attributes for discoverability.
*   NixOS modules should be placed in the `modules` attribute.
*   Overlays should be placed in the `overlays` attribute.
*   Reusable Nix functions should be placed in the `lib` attribute.

#### Providing NixOS modules

```nix
{ pkgs }: {
  modules = import ./modules;
}
```

```nix
# modules/default.nix
{
  example-module = ./example-module.nix;
}
```

A module with no [_class](https://nixos.org/manual/nixpkgs/stable/index.html#module-system-lib-evalModules-param-class) will be assumed to be both a NixOS and Home Manager module. If a module is NixOS or Home Manager specific, the `_class` attribute should be set to `"nixos"` or [`"home-manager"`](https://github.com/nix-community/home-manager/commit/26e72d85e6fbda36bf2266f1447215501ec376fd).

#### Providing Overlays

```nix
# default.nix
{
  overlays = {
    hello-overlay = import ./hello-overlay;
  };
}
```

```nix
# hello-overlay/default.nix
self: super: {
  hello = super.hello.overrideAttrs (old: {
    separateDebugInfo = true;
  });
}
```

#### Providing library functions

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

Override repositories using the `repoOverrides` argument.

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

### Overriding repositories with Flake

**Experimental**
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

## Contribution Guidelines

*   Ensure packages build and set `meta.broken = true` if they don't.
*   Supply standard [Nixpkgs meta attributes](https://nixos.org/nixpkgs/manual/#sec-standard-meta-attributes).
*   Keep repositories slim and leverage Nixpkgs packages whenever possible.

## Examples of Packages for NUR

*   Packages for a small audience.
*   Pre-releases.
*   Old package versions.
*   Automatic package generation.
*   Software with opinionated patches.
*   Experiments.

## Contact

*   Matrix: [#nur:nixos.org](https://matrix.to/#/#nur:nixos.org)
*   Discourse: [https://discourse.nixos.org/](https://discourse.nixos.org/)