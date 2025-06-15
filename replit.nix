
{ pkgs }: {
  deps = [
    pkgs.python311
    pkgs.python311Packages.pip
    pkgs.python311Packages.torch
    pkgs.python311Packages.transformers
    pkgs.python311Packages.datasets
    pkgs.python311Packages.pandas
    pkgs.python311Packages.seqeval
    pkgs.python311Packages.pytorch-crf
  ];
}
