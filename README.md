# kspec_pipe_controller

# K-SPEC Data Reduction Pipeline Controller
* The K-SPEC data reduction pipeline adopts [2dfdr](https://dev.aao.org.au/rds/2dfdr/2dfdr) software.

# Installation of 2dfdr
Try the steps outlined in the [2dfdr README](https://dev.aao.org.au/rds/2dfdr/2dfdr/-/tree/master?ref_type=heads#building-2dfdr) first:
```
make
make all install
```
<details>
<summary>These were what I had to do</summary>
<br >
<!-- ## 1. Conda environment
```
conda env create -n twodfdr --file=conda.yaml
conda activate twodfdr
```
## 2. Build 2dfdr -->
 - Replace a line in `2dfdr/acinclude.m4`
    ```m4
    AC_SUBST([subdirs_xp], ["$subdirs_xp^m4_normalize([$2])"])dnl
    ```
    with
    ```m4
    AC_SUBST([subdirs_xp], ["$subdirs_xp^CC=\"$CC\" CXX=\"$CXX\" FC=\"$FC\" CFLAGS=\"$CFLAGS\" CPPFLAGS=\"$CPPFLAGS\" CXXFLAGS=\"$CXXFLAGS\" LDFLAGS=\"$LDFLAGS\" m4_normalize([$2])"])dnl
    ])
    ```
    This is to avoid errors when building 2dfdr submodules with `gfortran`, `gcc`, `gxx` installed by conda.

 <!-- - Fix `update_2dfdr_verfile` in `2dfdr`:
    ```sh
    if [ $# != 1 ]; then
        echo "$0:Usage  $0 <filename>" 1>&2
        exit 1
    fi

    if git rev-parse --is-inside-work-tree > /dev/null 2>&1; then # this line
        env -u GIT_DIR git log --pretty --format="%H" -n1   >  $1
        env -u GIT_DIR git describe --dirty                 >> $1
        env -u GIT_DIR git show | head -n3 | tail -n2       >> $1
        env -u GIT_DIR git status                           >> $1
    else
        echo "$0: Not inside a Git working tree" 1>&2 # and this line
        exit 1
    fi
    ```
    This correction considers when `2dfdr` is a git submodule of another repository. The original script does not work in this case. -->

 <!-- - Configure `2dfdr`:
    ```sh
    ./configure
    ```
    If `configure` file does not exist, run:
    ```sh
    make
    ```
    `./configure` will be completed without errors, but `make` will fail with a compilation error in `2dfdr/pgplot_ac` submodule. -->

 - Comment out two lines in `2dfdr/pgplot_ac/GNUmakefile.am` and `2dfdr/pgplot_ac/GNUmakefile.in`, which contain the static library flags:
    ```makefile
    # AM_LDFLAGS = -static-libgfortran -static-libgcc # comment out these lines
    ```
    This is to avoid errors when building `pgplot` submodule with `gfortran`, `gcc`, `gxx` installed by conda.

<!-- Hmm… this isn’t working well.

Also, now that I’m continuously modifying the 2dfdr build, it feels pointless to keep 2dfdr as a submodule in this repository. If I’ve already decided to adjust the original 2dfdr code, then there’s little reason to keep trying to build KSPEC’s Python functions without touching it. -->

I successfully built 2dfdr with the root compiler, gcc=8.5.0, gxx=8.5.0, gfortran=8.5.0, with above modifications.
</details>

# Building the twodfdr Python module

## Construct a new conda environment
```sh
conda create -n twodfdr --file=conda_gcc14.yaml
conda activate twodfdr
```
## Install PyCPL and PyEsoRex
* [Building the twodfdr Python module](https://dev.aao.org.au/rds/2dfdr/2dfdr/-/tree/dev?ref_type=heads#building-the-twodfdr-python-module)
* [Installing PyCPL and PyEsoRex](https://www.eso.org/sci/software/pycpl/pycpl-site/user/installation.html#installing-pycpl)

