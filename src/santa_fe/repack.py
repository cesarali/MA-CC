"""Losslessly repack one sealed Santa Fe cell with Zstandard Parquet.

Original files and the original seal remain as local backups until the new
files and seal have been installed. An interrupted repack is rolled back on
the next invocation before validation or retry.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pyarrow.parquet as pq

from .cluster import _cell_dir, _load_manifest, _sha, _valid_seal, _json
from .config import load_config


NAMES=("rounds.parquet","micro.parquet")


def _rollback(root:Path)->None:
    backup_seal=root/"cell_complete.json.repack_backup"
    if not backup_seal.exists():return
    for name in NAMES:
        backup=root/(name+".repack_backup")
        if backup.exists():os.replace(backup,root/name)
        candidate=root/(name+".repack_candidate")
        if candidate.exists():candidate.unlink()
    os.replace(backup_seal,root/"cell_complete.json")


def repack_cell(config_path:str,cell_id:int)->dict:
    config=load_config(config_path)
    manifest=_load_manifest(config)
    root=_cell_dir(config,cell_id)
    lock=root/"repack.lock"
    descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    try:
        os.close(descriptor)
        _rollback(root)
        seal_path=root/"cell_complete.json"
        expected={"config_sha256":manifest["config_sha256"],"cell_id":cell_id,
                  "episodes":config.episodes,"round_rows":config.episodes*(config.params.rounds+1),
                  "simulator_sha256":manifest["simulator_sha256"]}
        if not _valid_seal(seal_path,expected):raise ValueError(f"unsealed cell {cell_id}")
        seal=json.loads(seal_path.read_text())
        if all(pq.ParquetFile(root/name).metadata.row_group(0).column(0).compression=="ZSTD"
               for name in NAMES):
            return {"cell_id":cell_id,"status":"already_zstd"}
        before=sum((root/name).stat().st_size for name in NAMES)
        checksums={}
        for name in NAMES:
            table=pq.read_table(root/name)
            candidate=root/(name+".repack_candidate")
            pq.write_table(table,candidate,compression="zstd",compression_level=6)
            if not pq.read_table(candidate).equals(table,check_metadata=True):
                raise ValueError(f"round-trip mismatch for {name}")
            checksums[name]=_sha(candidate)
        backup_seal=root/"cell_complete.json.repack_backup"
        backup_seal.write_bytes(seal_path.read_bytes())
        try:
            for name in NAMES:
                os.replace(root/name,root/(name+".repack_backup"))
                os.replace(root/(name+".repack_candidate"),root/name)
            seal["files"].update(checksums)
            seal["storage_codec"]="zstd_level_6_lossless"
            _json(seal_path,seal)
            if not _valid_seal(seal_path,expected):raise ValueError("new cell seal failed")
        except BaseException:
            _rollback(root)
            raise
        for name in NAMES:
            (root/(name+".repack_backup")).unlink()
        backup_seal.unlink()
        after=sum((root/name).stat().st_size for name in NAMES)
        return {"cell_id":cell_id,"status":"repacked","before_bytes":before,
                "after_bytes":after,"saved_bytes":before-after}
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",required=True)
    parser.add_argument("--cell-id",type=int,required=True)
    args=parser.parse_args()
    print(json.dumps(repack_cell(args.config,args.cell_id)))


if __name__=="__main__":main()
