# Nix User Repository (NUR): Expand Your Nix Package Options

Nix User Repository (NUR) is a community-driven meta-repository that expands your Nix package options by providing access to user-contributed packages. **Discover a vast library of community-built packages, modules, overlays, and library functions to enhance your NixOS experience.**  

[Go to the original repository](https://github.com/nix-community/NUR)

**Key Features:**

*   📦 **Community-Driven:** Access a wide range of packages contributed and maintained by the Nix community.
*   🔄 **Decentralized Package Sharing:** Easily share and discover new packages faster than through traditional channels.
*   ✅ **Automated Evaluation:** NUR automatically checks and validates repositories before integrating updates.
*   ⚙️ **Flexible Integration:** Integrate NUR with Flakes, `packageOverrides`, NixOS configurations, and Home Manager.
*   🔎 **Package Discovery:** Find packages through the NUR website or the `nur-combined` repository.
*   🛠️ **Easy Repository Addition:** Contribute your own packages by creating a repository with a `default.nix` file and submitting a pull request.

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

Add NUR to your `~/.config/nixpkgs/config.nix` (for login user) or `/etc/nixos/configuration.nix` (for NixOS):

```nix
{
  packageOverrides = pkgs: {
    nur = import (builtins.fetchTarball "https://github.com/nix-community/NUR/archive/main.tar.gz") {
      inherit pkgs;
    };
  };
}
```

### Pinning

Pin the version for faster and more reliable builds:

```nix
builtins.fetchTarball {
  url = "https://github.com/nix-community/NUR/archive/3a6a6f4da737da41e27922ce2cfacf68a109ebce.tar.gz";
  sha256 = "04387gzgl8y555b3lkz9aiw9xsldfg4zmzp930m62qw8zbrvrshd";
}
```

## How to Use

Install or use packages from NUR:

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

**Important:**  Always review packages before installation, as NUR packages are not subject to the same review process as Nixpkgs.

### Example: Using a Single Package in a Devshell (Flakes)

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

### NixOS Configuration

Use NUR modules and overlays in your NixOS configuration:

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
}
```

### Integrating with Home Manager

Integrate NUR modules into your Home Manager configuration:

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
*   [nur-combined](https://github.com/nix-community/nur-combined)

## Contributing Your Own Repository

1.  **Create a Repository:** Make a Git repository with a `default.nix` file at the root.
2.  **Package Definition:**  Packages should return a set of Nix derivations, taking the `pkgs` argument from Nixpkgs:

    ```nix
    { pkgs }:
    {
      hello-nur = pkgs.callPackage ./hello-nur {};
    }
    ```

3.  **Package Structure:** Create a directory (e.g., `hello-nur`) containing a `default.nix` to define the package.
4.  **Add Your Repository to NUR:**  Edit `repos.json` and add your repository details.
5.  **Update the Lockfile:** Run `./bin/nur format-manifest` and commit the changes, then create a Pull Request.
6.  **Update Lockfile (Alternative):** Use the update service at `https://nur-update.nix-community.org/update?repo=<your_repo_name>` after pushing updates.

### Other repository options

#### Using a different nix file as root expression

To use a different file instead of `default.nix` to load packages from, set the `file`
option to a path relative to the repository root:

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
#### Git submodules

To fetch git submodules in repositories set `submodules`:

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

## Overriding Repositories

Override repositories using the `repoOverrides` argument in `packageOverrides` or with Flakes (experimental).

## Contribution Guidelines

*   Ensure packages build successfully.
*   Use standard Nixpkgs `meta` attributes.
*   Keep repositories lean.
*   Reuse packages from Nixpkgs when appropriate.

## Examples of Packages Suitable for NUR

*   Packages for a niche audience
*   Pre-releases
*   Older package versions
*   Automatically generated package sets
*   Software with custom patches
*   Experiments

## Contact

Join the conversation:

*   Matrix: [#nur:nixos.org](https://matrix.to/#/#nur:nixos.org)
*   Discourse: [https://discourse.nixos.org/](https://discourse.nixos.org/)