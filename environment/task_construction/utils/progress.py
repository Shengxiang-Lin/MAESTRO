"""Shared tqdm-based progress helpers for long-running environment commands."""
from __future__ import annotations
import gzip,io,sys
from pathlib import Path
from typing import Iterator,TextIO
from tqdm.auto import tqdm
class Progress:
    def __init__(self,label:str,*,total:int|None=None,unit:str="records",enabled:bool=False,stream:TextIO|None=None)->None:
        self._label=str(label); self.total=total; self.unit=unit; self.enabled=bool(enabled); self.stream=stream if stream is not None else sys.stderr; self.completed=0; self._bar=None
    @property
    def label(self): return self._label
    @label.setter
    def label(self,value):
        self._label=str(value)
        if self._bar is not None:self._bar.set_description_str(self._label,refresh=False)
    def __enter__(self):
        if self.enabled:
            byte_unit=self.unit=="B"; self._bar=tqdm(total=self.total,desc=self._label,unit=self.unit,unit_scale=byte_unit,unit_divisor=1024 if byte_unit else 1000,file=self.stream,disable=False,leave=True,dynamic_ncols=True)
        return self
    def update(self,completed:int)->None:
        completed=max(0,int(completed)); completed=min(completed,self.total) if self.total is not None else completed; delta=completed-self.completed; self.completed=completed
        if not self.enabled or self._bar is None:return
        if delta>=0:self._bar.update(delta)
        else:self._bar.n=completed; self._bar.refresh()
    def advance(self,amount:int=1)->None:self.update(self.completed+int(amount))
    def __exit__(self,exc_type,exc,traceback)->None:
        if not self.enabled or self._bar is None:return
        if exc_type is None:
            if self.total is not None and self.completed<self.total:self.update(self.total)
            self.label=f"{self._label} done"
        else:self.label=f"{self._label} failed"
        self._bar.close()
def iter_file_lines(path:str|Path,*,compressed:bool=False,progress:bool=False,description:str|None=None,errors:str="strict")->Iterator[str]:
    path=Path(path); label=description or path.name; label += " (compressed bytes)" if compressed else ""
    with path.open("rb") as raw:
        binary=gzip.GzipFile(fileobj=raw,mode="rb") if compressed else raw
        with io.TextIOWrapper(binary,encoding="utf-8",errors=errors) as handle:
            with Progress(label,total=path.stat().st_size if progress else None,unit="B",enabled=progress) as bar:
                for index,line in enumerate(handle,1):
                    yield line
                    if progress and index%4096==0:bar.update(raw.tell())
