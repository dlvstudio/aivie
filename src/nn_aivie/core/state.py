"""Graph state: vocabulary + adjacency."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

from nn_aivie.core.adjacency import TypedAdjacency
from nn_aivie.core.embedding import GraphEmbedding
from nn_aivie.core.parent_store import ParentStore
from nn_aivie.core.vocab import Vocabulary


@dataclass
class GraphState:
    vocab: Vocabulary = field(default_factory=Vocabulary)
    adjacency: TypedAdjacency = field(default_factory=TypedAdjacency)
    # E — graph-derived soft embeddings (view dẫn xuất của A, tính lại offline). Optional:
    # checkpoint cũ không có; build bằng core.embedding.build_embedding khi cần.
    embedding: GraphEmbedding | None = None
    # Nơ-ron cha (v3 §1) — "1 ví dụ = 1 nơ-ron, rồi phân rã". Lõi pattern-completion.
    parents: ParentStore | None = None
    # Mẫu câu mở lửng (sidecar) — KHÔNG add vào ParentStore (tránh dilution phrase index).
    pattern_templates: list[list[int]] = field(default_factory=list)
    # TK_042 G10 — LM nơ-ron làm kênh trôi chảy. KHÔNG serialize (là artifact riêng,
    # `out/g7_best_model.pt`); nạp lazy qua `HCAConfig.ar_lm_path`. None = kênh W_seq như cũ.
    neural_lm: object | None = None
    # TK_044 P3.1 — binding: cạnh coreference + role-backoff. sig = (left_ctx, right_ctx) tuple
    # token-id (entity→-1 sentinel). dict token-id = TRỌNG SỐ nơ-ron (như adjacency), KHÔNG text
    # → Luật 0.0 sạch. Học ở `training/coref_pass.run_coref_pass`, đọc lúc sinh (P3.2).
    coref: dict = field(default_factory=dict)         # sig → antecedent-sig (đồng-chỉ)
    after_role: dict = field(default_factory=dict)    # sig → token-id cấu-trúc kế (role-backoff)

    def __setattr__(self, name, value):
        """Gán `parents` thì đặt luôn BACKREF `ParentStore → GraphState` (weakref).

        Vì sao cần: `rank_matches` chỉ nhận `query_ids`/`vocab`, nhưng phá-thế-hoà Hopfield
        (TK_045 §13.9/§12.151) cần `state` để dùng lại `ParentBipartite.A` đã cache. Có
        **17 call-site** `rank_matches` — thêm tham số vào từng chỗ là churn lớn và dễ sót.
        Đặt ở ĐÂY thì mọi đường gán (`GraphState.load`, ingest, test) đều được phủ, một chỗ.

        **weakref** chứ không phải tham chiếu cứng: `state → parents → state` là chu trình,
        giữ cứng thì GC không thu được state cũ, mà đây đúng là repo từng dính lỗi cache
        theo `id(state)` khi state cũ bị thu hồi rồi id tái dùng (§12.29/§12.30).
        """
        object.__setattr__(self, name, value)
        if name == "parents" and value is not None:
            import weakref

            try:
                object.__setattr__(value, "_owner_ref", weakref.ref(self))
            except (AttributeError, TypeError):
                pass  # đối tượng không weakref/setattr được → bỏ qua, chỉ mất tie-break

    def save(self, path: str | Path) -> None:
        path = Path(path)
        is_json = path.suffix == ".json"
        payload = {
            "vocab_tokens": self.vocab._id_to_token,
            # Pickle: mảng numpy (load nhanh). JSON: key chuỗi (khả chuyển). from_dict nhận cả 2.
            "adjacency": self.adjacency.to_dict() if is_json else self.adjacency.to_arrays(),
        }
        if self.vocab.tokenizer is not None:
            # TK_041: checkpoint MANG THEO tokenizer của chính nó (~1MB). Đắt hơn ghim
            # fingerprint, nhưng loại HẲN lớp lỗi "tokenizer lệch checkpoint" — lớp lỗi đó
            # không crash, nó chỉ ra rác LẶNG LẼ (mọi id trỏ sai token). Self-contained.
            payload["tokenizer"] = self.vocab.tokenizer.to_payload()
        if self.embedding is not None:
            payload["embedding"] = self.embedding.to_payload()
        if self.parents is not None:
            payload["parents"] = self.parents.to_payload()
        if self.pattern_templates:
            payload["pattern_templates"] = self.pattern_templates
        if self.coref:  # TK_044 P3.1 — sig lồng (left,right) → list để pickle/json ổn định
            payload["coref"] = [
                [[list(a[0]), list(a[1])], [list(b[0]), list(b[1])]] for a, b in self.coref.items()
            ]
        if self.after_role:
            payload["after_role"] = [[[list(k[0]), list(k[1])], v] for k, v in self.after_role.items()]
        if is_json:
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            with path.open("wb") as f:
                pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str | Path, *, logger=None, freeze_adjacency: bool = False) -> GraphState:
        """`logger` (FileProgressLog) → ghi tiến độ TỪNG BƯỚC load ra file (quan sát được:
        load đến đâu, đúng chưa). None = im lặng (test/inference tight-loop)."""
        from contextlib import nullcontext

        path = Path(path)

        def _phase(name: str):
            return logger.phase(name) if logger is not None else nullcontext()

        if logger is not None:
            logger.log(f"load checkpoint {path} …")
        with _phase("unpickle"):
            if path.suffix == ".json":
                payload = json.loads(path.read_text(encoding="utf-8"))
            else:
                with path.open("rb") as f:
                    payload = pickle.load(f)

        state = cls()
        with _phase(f"vocab ({len(payload['vocab_tokens'])} token)"):
            tok_payload = payload.get("tokenizer")
            if tok_payload is not None:
                from nn_aivie.core.tokenizer import BPETokenizer

                state.vocab = Vocabulary(BPETokenizer.from_payload(tok_payload))
            # Replay theo THỨ TỰ đã lưu → id giữ nguyên. Với vocab BPE, dải bề mặt đã được
            # `Vocabulary.__init__` nạp sẵn đúng thứ tự đó, nên `id_of` ở đây chỉ tra lại;
            # phần đuôi (Entity:/Concept:/…) mới thật sự thêm.
            for token in payload["vocab_tokens"]:
                state.vocab.id_of(token)
            if tok_payload is not None and state.vocab._id_to_token != payload["vocab_tokens"]:
                raise ValueError(
                    "vocab replay lệch tokenizer — checkpoint và merge table không cùng đời"
                )
        with _phase("adjacency"):
            state.adjacency = TypedAdjacency.from_dict(
                payload["adjacency"], csr=freeze_adjacency
            )
            if logger is not None and freeze_adjacency:
                logger.log("    adjacency: CSR frozen (TK_040 low-peak)")
        if payload.get("embedding") is not None:
            with _phase("embedding"):
                state.embedding = GraphEmbedding.from_payload(payload["embedding"])
        if payload.get("parents") is not None:
            with _phase("parents"):
                state.parents = ParentStore.from_payload(payload["parents"], logger=logger)
                # `match_ids` serialize đóng băng luật `_is_match_token` LÚC TRAIN — dựng lại
                # từ `slot_chain` để thống kê (_gdf → IDF) luôn khớp luật HIỆN TẠI. Không có
                # bước này, checkpoint train trước 2026-07-15 giữ `_gdf['.']`=0 → gidf kịch
                # trần → dấu chấm thành neo (không còn is_punct chặn). Xem TK_012 §9.
                state.parents.resync_match_ids(state.vocab, logger=logger)
        if payload.get("pattern_templates"):
            state.pattern_templates = payload["pattern_templates"]
        if payload.get("coref"):  # TK_044 P3.1 — dựng lại sig tuple lồng (left,right)
            def _sig(x):
                return (tuple(x[0]), tuple(x[1]))
            state.coref = {_sig(a): _sig(b) for a, b in payload["coref"]}
        if payload.get("after_role"):
            state.after_role = {(tuple(k[0]), tuple(k[1])): v for k, v in payload["after_role"]}
        if logger is not None:
            logger.log(
                f"✓ loaded: vocab={len(state.vocab)} "
                f"parents={len(state.parents.parents) if state.parents else 0} "
                f"E={state.embedding.matrix.shape if state.embedding else None}"
            )
        return state
