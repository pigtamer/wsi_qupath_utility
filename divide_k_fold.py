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
                        default="./TILES/")

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
    execlevel(src_dir="./sample_directory/",
              func=create_folds,
              N=5,
              depth=2)

    return 0


if __name__ == '__main__':
    main()
