#!/usr/bin/env python

import os
import tarfile

import click
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
from allennlp.models.archival import CONFIG_NAME, _WEIGHTS_NAME



@click.command()
@click.argument('config_file')
@click.argument('serialization_dir')
@click.option('--weights-file', default="best.th")
@click.option('--archive-name', default="model.tar.gz")
@click.option('--exist-ok', default=False)
def generate_archive(
        config_file: str,
        serialization_dir: str,
        weights_file: str = "best.th",
        archive_name: str = "model.tar.gz",
        exist_ok: bool = False,
) -> None:
    archive_file = os.path.join(serialization_dir, archive_name)

    if os.path.exists(archive_file):
        if exist_ok:
            print("removing archive file %s" % archive_file)
        else:
            print("archive file %s already exists" % archive_file)
            sys.exit(-1)

    print("creating new archive file %s" % archive_file)

    with tarfile.open(archive_file, "w:gz") as archive:
        archive.add(config_file, arcname=CONFIG_NAME)
        archive.add(os.path.join(serialization_dir, weights_file), arcname=_WEIGHTS_NAME)
        archive.add(os.path.join(serialization_dir, "vocabulary"), arcname="vocabulary")


if __name__ == "__main__":
    generate_archive()
