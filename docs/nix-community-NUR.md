# NUR: The Nix User Repository

**Expand your Nix package library with community-contributed packages through NUR, a decentralized, user-driven repository.**

NUR (Nix User Repository) empowers the Nix community by providing a platform to share and discover user-contributed packages, modules, and overlays, expanding the available software options beyond Nixpkgs.  Developed by the Nix community, NUR allows users to quickly access new and specialized packages built from source, promoting faster community-driven package distribution.

## Key Features

*   **Community-Driven:** Access packages contributed and maintained by the Nix community.
*   **Decentralized:**  Easily share and find packages not yet in Nixpkgs.
*   **Flexible Package Management:** Integrate NUR packages into your Nix configurations using flakes, package overrides, and more.
*   **Automated Checks:** Automatic evaluation checks before updates ensure package integrity.
*   **Modules, Overlays, & Library Functions:** Supports NixOS modules, overlays, and library functions, enhancing customization.

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

Then use either `overlays.default` or `legacyPackages.<system>`.

### Using `packageOverrides`

Add the following to `~/.config/nixpkgs/config.nix`:

```nix
{
  packageOverrides = pkgs: {
    nur = import (builtins.fetchTarball "https://github.com/nix-community/NUR/archive/main.tar.gz") {
      inherit pkgs;
    };
  };
}
```

For NixOS, add the following to your `/etc/nixos/configuration.nix`

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

To prevent frequent downloads, pin the version of NUR:

```nix
builtins.fetchTarball {
  url = "https://github.com/nix-community/NUR/archive/3a6a6f4da737da41e27922ce2cfacf68a109ebce.tar.gz";
  sha256 = "04387gzgl8y555b3lkz9aiw9xsldfg4zmzp930m62qw8zbrvrshd";
}
```

## How to Use

Install packages from the NUR namespace using `nix-shell`, `nix-env`, or in your NixOS configuration.

```bash
$ nix-shell -p nur.repos.mic92.hello-nur
$ nix-env -f '<nixpkgs>' -iA nur.repos.mic92.hello-nur
```

Or in `configuration.nix`:

```nix
environment.systemPackages = with pkgs; [
  nur.repos.mic92.hello-nur
];
```

***Important:*** *Always verify expressions before installing, as NUR packages are not reviewed by Nixpkgs maintainers.*

### Using a single package in a devshell (Flakes Example)

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
          overlays = [ nur.overlays.default ];
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

### Using NUR in NixOS (Modules Example)

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
        # Example overlay
      ];
    };
  };
}
```

### Integrating with Home Manager

Integrate modules by adding them to `imports`:

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

*   Browse packages using the [NUR Packages Search](https://nur.nix-community.org/).
*   Explore the [nur-combined](https://github.com/nix-community/nur-combined) repository on GitHub.

## Contributing Your Own Repository

1.  Create a repository with a `default.nix` file at its root, using the [repository template](https://github.com/nix-community/nur-packages-template).
2.  Avoid importing packages directly from `<nixpkgs>`. Instead, use the `pkgs` argument provided to your repository.
3.  Each repository should return a set of Nix derivations.
    ```nix
    { pkgs }:
    {
      hello-nur = pkgs.callPackage ./hello-nur {};
    }
    ```
4.  Add your repository to NUR's `repos.json`.

    ```bash
    $ git clone --depth 1 https://github.com/nix-community/NUR
    $ cd NUR
    ```
5.  Edit `repos.json`:

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

6.  Run `./bin/nur format-manifest` and add the changes, commit, and push. Then create a pull request.

### Using a different nix file as root expression

Set the `file` option to a path relative to the repository root:

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

### Updating NUR's Lockfile

Use the service at  [https://nur-update.nix-community.org/](https://nur-update.nix-community.org/) to update NUR faster after you have pushed changes to your repository.

### Troubleshooting Package Updates

Ensure your repository's evaluation succeeds to update packages. Common causes include:

*   Incorrect license attributes in metadata.
*   Using built-in fetchers instead of `pkgs.fetch*`.

Check the [latest build job](https://github.com/nix-community/NUR/actions) to determine the evaluation state.

#### Local Evaluation Check

Run the [check evaluation](https://github.com/nix-community/nur-packages-template/blob/main/.github/workflows/build.yml) task in your `nur-packages/` folder.

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

To fetch git submodules, set `submodules`:

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

### NixOS Modules, Overlays and Library Function Support

Structure your repository to support NixOS modules, overlays and library functions.

#### Providing NixOS modules

Place NixOS modules in the `modules` attribute:

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

Use the `overlays` attribute:

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

Use the `lib` attribute:

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

You can override repositories using `repoOverrides`.

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

### Overriding Repositories with Flake (Experimental)

-   With packageOverrides
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

-   With overlay
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

The repo must contain a `flake.nix` file in addition to a `default.nix`:  [flake.nix example](https://github.com/Mic92/nur-packages/blob/master/flake.nix)

## Contribution Guidelines

*   Ensure packages build and set `meta.broken` to `true` if broken.
*   Provide standard [Nixpkgs meta attributes](https://nixos.org/nixpkgs/manual/#sec-standard-meta-attributes).
*   Keep repositories lean to optimize download and evaluation.
*   Reuse packages from Nixpkgs when possible.

## Examples of Packages for NUR

*   Packages for a small audience.
*   Pre-releases.
*   Older package versions.
*   Automated package generation (PyPI, CPAN).
*   Software with custom patches.
*   Experiments.

## Get Involved

Join the discussion and get help:

*   Matrix: [#nur:nixos.org](https://matrix.to/#/#nur:nixos.org)
*   Discourse: [https://discourse.nixos.org/](https://discourse.nixos.org/)

[Back to Top](#top)

[Original Repository](https://github.com/nix-community/NUR)