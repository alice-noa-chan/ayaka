"""Use POSIX remote paths with Beam's existing multipart CLI on Windows."""

import posixpath


def remote_path(remote):
    return posixpath.join(remote.volume_name, remote.volume_path.replace("\\", "/"))


if __name__ == "__main__":
    from beam.cli.main import cli
    from beta9.multipart import RemotePath

    # SDK 0.2.207 uses os.path.join here, sending backslashes to the Linux server.
    # Scope the compatibility patch to this transfer process; local paths stay native.
    RemotePath.path = property(remote_path)
    cli()
