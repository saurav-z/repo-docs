# NUR: The Nix User Repository

**Expand your Nix ecosystem with community-driven packages and configurations, all built from source!**  [Learn more about NUR on GitHub](https://github.com/nix-community/NUR).

## Key Features

*   **Community-Driven:** Access a vast collection of user-contributed packages and configurations.
*   **Decentralized Package Sharing:** Easily share and discover new packages faster than traditional repositories.
*   **Flexible Installation:** Integrate NUR with flakes, `packageOverrides`, NixOS configurations, and Home Manager.
*   **Automated Checks:** NUR automatically validates repositories and performs evaluation checks to ensure stability.
*   **Package Discovery:** Find packages through a search interface ([Packages search for NUR](https://nur.nix-community.org/)) and the `nur-combined` repository.
*   **Extensible:** Support for NixOS modules, overlays, and library functions.

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

Use the overlay (`overlays.default`) or `legacyPackages.<system>`.

### Using `packageOverrides`

Add to `~/.config/nixpkgs/config.nix`:

```nix
{
  packageOverrides = pkgs: {
    nur = import (builtins.fetchTarball "https://github.com/nix-community/NUR/archive/main.tar.gz") {
      inherit pkgs;
    };
  };
}
```

For NixOS, add to `/etc/nixos/configuration.nix` (and `~/.config/nixpkgs/config.nix` if you use NUR in `nix-env`, home-manager, or `nix-shell`).

### Pinning

Pin versions for faster builds:

```nix
builtins.fetchTarball {
  url = "https://github.com/nix-community/NUR/archive/3a6a6f4da737da41e27922ce2cfacf68a109ebce.tar.gz";
  sha256 = "04387gzgl8y555b3lkz9aiw9xsldfg4zmzp930m62qw8zbrvrshd";
}
```

## How to Use

Install packages:

```bash
$ nix-shell -p nur.repos.mic92.hello-nur
```

```bash
$ nix-env -f '<nixpkgs>' -iA nur.repos.mic92.hello-nur
```

Or configure in NixOS:

```nix
# configuration.nix
environment.systemPackages = with pkgs; [
  nur.repos.mic92.hello-nur
];
```

**Important:** Always review packages before installation.

### Example: Devshell with a Single Package

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

### Example: Using NUR in NixOS

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
*   [nur-combined](https://github.com/nix-community/nur-combined/search)

## Adding Your Own Repository

1.  Create a Git repository with a `default.nix` file.  A [repository template](https://github.com/nix-community/nur-packages-template) is available.
2.  Use `pkgs` as the argument, instead of `import <nixpkgs> {}`.
3.  Return a set of Nix derivations.
4.  Example `default.nix`:

    ```nix
    { pkgs }:
    {
      hello-nur = pkgs.callPackage ./hello-nur {};
    }
    ```

    Example `hello-nur/default.nix`:

    ```nix
    { stdenv, fetchurl, lib }:

    stdenv.mkDerivation rec {
      pname = "hello";
      version = "2.10";

      src = fetchurl {
        url = "mirror://gnu/hello/${pname}-${version}.tar.gz";
        sha256 = "0ssi1wpaf7plaswqqjwigppsg5fyh99vdlb9kzl7c9lng89ndq1i";
      };

      postPatch = ''
        sed -i -e 's/Hello, world!/Hello, NUR!/' src/hello.c
      '';

      doCheck = false;

      meta = with lib; {
        description = "A program that produces a familiar, friendly greeting";
        longDescription = ''
          GNU Hello is a program that prints "Hello, world!" when you run it.
          It is fully customizable.
        '';
        homepage = https://www.gnu.org/software/hello/manual/;
        changelog = "https://git.savannah.gnu.org/cgit/hello.git/plain/NEWS?h=v${version}";
        license = licenses.gpl3Plus;
        maintainers = [ maintainers.eelco ];
        platforms = platforms.all;
      };
    }
    ```
5.  Build your packages with `nix-shell` or `nix-build`.
6.  Add your repository to `repos.json` in the NUR repository:

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
            "<fill-your-repo-name>": {
                "url": "https://github.com/<your-user>/<your-repo>"
            }
        }
    }
    ```

    Run:

    ```bash
    $ ./bin/nur format-manifest
    $ git add repos.json
    $ git commit -m "add <your-repo-name> repository"
    $ git push
    ```

7.  Open a pull request to [https://github.com/nix-community/NUR](https://github.com/nix-community/NUR).

### Using a Different Root Expression

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

### Updating the Lock File

Use the nur-update service after updating your repository: `curl -XPOST https://nur-update.nix-community.org/update?repo=mic92`

### Troubleshooting Package Updates

*   Check the [latest build job](https://github.com/nix-community/NUR/actions) for evaluation errors.
*   Common errors:  wrong license or using `builtins.fetchGit` instead of `pkgs.fetchgit`.

#### Local Evaluation Check

In your `nur-packages/` folder, run the [check evaluation](https://github.com/nix-community/nur-packages-template/blob/main/.github/workflows/build.yml) task

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

To fetch submodules, set `submodules`:

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

Organize these within your repository:

*   **NixOS Modules:** Place in the `modules` attribute:

    ```nix
    { pkgs }: {
      modules = import ./modules;
    }
    ```

    Modules should be defined as paths, not functions. A module with no `_class` will be assumed to be both a NixOS and Home Manager module. If a module is NixOS or Home Manager specific, the `_class` attribute should be set to `"nixos"` or [`"home-manager"`](https://github.com/nix-community/home-manager/commit/26e72d85e6fbda36bf2266f1447215501ec376fd).
*   **Overlays:**  Use the `overlays` attribute.
*   **Library Functions:** Place in the `lib` attribute.

## Overriding Repositories

Use the `repoOverrides` argument:

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

### Overriding Repositories with Flakes

Use `packageOverrides` or an overlay.  The repo must contain a `flake.nix` file.

## Contribution Guidelines

*   Build packages and set `meta.broken` to `true` if they don't.
*   Supply meta attributes (see the [Nixpkgs manual](https://nixos.org/nixpkgs/manual/#sec-standard-meta-attributes)).
*   Keep repositories slim.
*   Reuse packages from Nixpkgs when possible.

## Use Cases

*   Packages for a small audience.
*   Pre-releases.
*   Old versions of packages.
*   Automatically generated package sets.
*   Software with custom patches.
*   Experiments.

## Contact

*   Matrix: [#nur:nixos.org](https://matrix.to/#/#nur:nixos.org)
*   Discourse: [https://discourse.nixos.org/](https://discourse.nixos.org/)