# To learn more about how to use Nix to configure your environment
# see: https://firebase.google.com/docs/studio/customize-workspace
{ pkgs, ... }: {
  # Which nixpkgs channel to use.
  channel = "stable-24.05"; # or "unstable"
  # Use https://search.nixos.org/packages to find packages
  packages = [
    pkgs.docker
    pkgs.docker-compose
  ];
  # Sets environment variables in the workspace
  env = {
    XDG_RUNTIME_DIR="/tmp/runtime-dir";
    DOCKER_HOST="unix:///tmp/runtime-dir/docker.sock";
  };
  idx = {
    # Search for the extensions you want on https://open-vsx.org/ and use "publisher.id"
    extensions = [
      "vscodevim.vim"
    ];
    workspace = {
      # Runs when a workspace is first created with this `dev.nix` file
      onCreate = {
        # Open editors for the following files by default, if they exist:
      };
      onStart = {
        dockerd-rootless = "mkdir -p $XDG_RUNTIME_DIR && dockerd-rootless &";
      };
      # To run something each time the workspace is (re)started, use the `onStart` hook
    };
      # Enable previews and customize configuration
    previews = {
      enable = false;
    };
  };
}

