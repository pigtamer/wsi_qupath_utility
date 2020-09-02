"""
Divide the dataset into k folds (e.g. 10)
本文件生成k折交叉验证的目录结构

Tiles
    Blood
    Tumor
    ...

>>>

Tiles
    Blood
        01
        02
        ...
        10
    Tumor
        01
        02
        ...
        10
    ...


"""

import argparse
import os
import shutil


def create_folds(dirname, N=10):
    """Move files into N subdirectories.
    N: directory number
    divide equally (except the last directory)
    """
    abs_dirname = os.path.abspath(dirname)
    files = [os.path.join(abs_dirname, f.path)
             for f in os.scandir(abs_dirname) if not f.is_dir()]

    i = 0
    file_num = len(files)
    num_in_subdir = file_num//N + 1
    for k in range(N):
        subdir_name = os.path.join(
            abs_dirname, '{0:03d}'.format(k+1))
        if not os.path.exists(subdir_name):
            os.mkdir(subdir_name)
    k = 0
    for f in sorted(files):
        if i % num_in_subdir != 0:
            pass
        else:
            k += 1
        subdir_name = os.path.join(
            abs_dirname, '{0:03d}'.format(k))
        # move file to current dir
        f_base = os.path.basename(f)
        shutil.move(f, os.path.join(subdir_name, f_base))
        i += 1


def unfold(dirname, N=None):
    """move files in subdirs (folds) out to the basedir
    dirname: base directory
    """
    dirname = os.path.abspath(dirname)
    subfolders = [f.path for f in os.scandir(dirname) if (
        f.is_dir() and not os.path.basename(f).startswith('.'))]
    for subdir in subfolders:
        abs_dirname = os.path.abspath(subdir)
        files = [os.path.join(abs_dirname, f) for f in os.listdir(abs_dirname)]
        for f in sorted(files):
            f_base = os.path.basename(f)
            shutil.move(f, os.path.join(dirname, f_base))
        shutil.rmtree(abs_dirname)


def parse_args():
    """Parse command line arguments passed to script invocation."""
    parser = argparse.ArgumentParser(
        description='Split files into multiple subfolders.')

    parser.add_argument('src_dir',
                        help='source directory',
                        type=str,
                        default="./TILES/")
    parser.add_argument('depth',
                        help='depth of diectory operations',
                        type=int,
                        default=100)
    parser.add_argument('k',
                        help='fold number',
                        type=int,
                        default=10)
    parser.add_argument('mode',
                        type=int,
                        help='create fold/undo',
                        default=None)

    return parser.parse_args()


def execlevel(basedir, func=None, N=None, depth=2):
    """execute fold or unfold to a specified level of directory

    Args:
        basedir ([type]): [description]
        func ([type], optional): [description]. Defaults to None.
        N ([type], optional): [description]. Defaults to None.
        depth (int, optional): [description]. Defaults to 2.
    """
    curr_dirs = [f.path for f in os.scandir(basedir) if (
        f.is_dir() and not os.path.basename(f).startswith('.'))]
    if depth == 1:
        for subdir in curr_dirs:
            func(subdir, N)
    else:
        depth -= 1
        for subdir in curr_dirs:
            execlevel(subdir, func=func, N=N, depth=depth)


def main():
    args = parse_args()
    src_dir = args.src_dir  # "./sample_directory/"

    # Walk till the last hierachy of directory
    if args.mode == 0:
        func = unfold
    elif args.mode == 1:
        func = create_folds
    else:
        print(args.mode)
        return 1

    execlevel(basedir=src_dir,
              func=func,
              N=args.k,
              depth=args.depth)

    return 0


if __name__ == '__main__':
    main()


""" 
.
└── mask
    ├── annot
    │   └── fooannot.annot
    └── tile
        ├── healthy
        │   ├── 001 [27 entries exceeds filelimit, not opening dir]
        │   ├── 002 [27 entries exceeds filelimit, not opening dir]
        │   ├── 003 [27 entries exceeds filelimit, not opening dir]
        │   ├── 004 [27 entries exceeds filelimit, not opening dir]
        │   └── 005 [24 entries exceeds filelimit, not opening dir]
        └── tumor
            ├── 001 [21 entries exceeds filelimit, not opening dir]
            ├── 002 [21 entries exceeds filelimit, not opening dir]
            ├── 003 [21 entries exceeds filelimit, not opening dir]
            ├── 004 [21 entries exceeds filelimit, not opening dir]
            └── 005 [17 entries exceeds filelimit, not opening dir]

python ../divide_k_fold.py ./mask/tile/ 1 5 0
    ===
        python ../divide_k_fold.py ./ 3 5 0

===>

.
└── chips(same structure)
........
└── mask
    ├── annot
    │   └── fooannot.annot
    └── tile
        ├── healthy [132 entries exceeds filelimit, not opening dir]
        └── tumor [101 entries exceeds filelimit, not opening dir]

 """
