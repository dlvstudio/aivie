from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pycrfsuite

from nn_aivie.ws_crf.ws_features import WSFeatureConfig, build_ws_features


@dataclass
class WSCRFModel:
    """Word segmentation CRF (B-W/I-W) — offline/build tool."""

    tagger: pycrfsuite.Tagger
    dictionary_lower: set[str]
    cfg: WSFeatureConfig = WSFeatureConfig()

    @classmethod
    def load(cls, model_dir: str | Path) -> "WSCRFModel":
        model_dir = Path(model_dir)
        meta = json.loads((model_dir / "meta.json").read_text("utf-8"))
        dict_path = model_dir / meta.get("dictionary_file", "dictionary_lower.txt")
        dictionary_lower = set(
            w.strip() for w in dict_path.read_text("utf-8").splitlines() if w.strip()
        )
        tagger = pycrfsuite.Tagger()
        tagger.open(str(model_dir / meta.get("model_file", "model.crfsuite")))
        cfg = WSFeatureConfig(window=int(meta.get("window", 2)))
        return cls(tagger=tagger, dictionary_lower=dictionary_lower, cfg=cfg)

    def tag(self, tokens: list[str]) -> list[str]:
        xseq = [
            build_ws_features(tokens, i, dictionary_lower=self.dictionary_lower, cfg=self.cfg)
            for i in range(len(tokens))
        ]
        return list(self.tagger.tag(xseq))

    def merge_iw(self, tokens: list[str], tags: list[str]) -> list[str]:
        out: list[str] = []
        for t, tok in zip(tags, tokens):
            if t == "I-W" and out:
                out[-1] = out[-1] + " " + tok
            else:
                out.append(tok)
        return out

