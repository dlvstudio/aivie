"""CLI entry point for aivie."""

from __future__ import annotations

import argparse
import sys

from nn_aivie.core.paths import default_checkpoint


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(prog="aivie", description="AIVIE — Hebbian SLM + curriculum")
    sub = parser.add_subparsers(dest="command", required=True)

    gen_p = sub.add_parser("generate", help="Generate tokens from checkpoint")
    gen_p.add_argument("--prompt", required=True, help="Prompt text")
    gen_p.add_argument(
        "--checkpoint",
        default=str(default_checkpoint()),
        help="Graph state checkpoint",
    )
    gen_p.add_argument("--max-len", type=int, default=32)
    gen_p.add_argument(
        "--mode",
        choices=["w_skip", "frame_da_khe", "compose_2cha", "analogy_la"],
        default="w_skip",
        help=(
            "w_skip = mặc định (giữ eval/legacy). "
            "frame_da_khe = 1 khung + vá P112 (RUNBOOK_FRAME_P112_SHIP.md). "
            "compose_2cha = ghép ≥2 cha P203 (RUNBOOK_COMPOSE_P203_SHIP.md). "
            "analogy_la = điền X là ? P98G (RUNBOOK_ANALOGY_P98G_SHIP.md; mini OK). "
            "frame_da_khe/compose_2cha cần checkpoints/network_v12_subst.pkl."
        ),
    )
    gen_p.add_argument(
        "--trace-log",
        action="store_true",
        help="Ghi trace từng token (w0, token chọn, agenda, top scores) ra out/logs/",
    )

    train_p = sub.add_parser(
        "train",
        help="Corpus training theo lớp + giai đoạn G1–G4 (nanochat-style, có manifest)",
    )
    train_p.add_argument("--grade", type=int, required=True, help="Lớp curriculum (1–12)")
    train_p.add_argument(
        "--phase",
        choices=["G1", "G2", "G3", "G4"],
        default="G1",
        help="Giai đoạn học tập trong lớp",
    )
    train_p.add_argument("--checkpoint", default=None, help="Load graph state trước khi train")
    train_p.add_argument(
        "-o",
        "--output",
        default=str(default_checkpoint()),
        help="Checkpoint output",
    )
    train_p.add_argument("--eta", type=float, default=None, help="η Hebbian (G1/G3)")
    train_p.add_argument(
        "--alpha",
        type=float,
        default=1.5,
        help="Hệ số thưởng dopamine G3 (mặc định 1.5)",
    )
    train_p.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Hệ số phạt LTD G3 (mặc định 1.0)",
    )
    train_p.add_argument(
        "--no-teach-eos",
        action="store_true",
        help="G3: không thêm bước học <EOS> sau mỗi câu",
    )
    train_p.add_argument(
        "--no-sentence-bind",
        action="store_true",
        help="Tắt co-activate toàn câu (chỉ cửa sổ local)",
    )
    train_p.add_argument(
        "--sentence-eta-scale",
        type=float,
        default=0.5,
        help="Hệ số η khi bind toàn câu (mặc định 0.5 × eta)",
    )
    train_p.add_argument("--max-records", type=int, default=None, help="Giới hạn số câu")
    train_p.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="Chỉ train subset dataset id (lặp lại flag để nhiều id)",
    )
    train_p.add_argument("--manifest", default=None, help="Đường dẫn manifest.yaml tùy chọn")

    export_viz_p = sub.add_parser(
        "export-viz",
        help="Export subgraph JSON (+ optional trace) cho Graph UI",
    )
    export_viz_p.add_argument(
        "--checkpoint",
        default=str(default_checkpoint()),
        help="GraphState checkpoint",
    )
    export_viz_p.add_argument(
        "--rules",
        nargs="+",
        default=["vi_QA_hoi_dap", "vi_copula_S_la_NA"],
    )
    export_viz_p.add_argument("--min-weight", type=float, default=0.15)
    export_viz_p.add_argument("--max-edges", type=int, default=800)
    export_viz_p.add_argument("--hops", type=int, default=2)
    export_viz_p.add_argument(
        "-o",
        "--output",
        default="viz/web/public/graph.json",
    )
    export_viz_p.add_argument("--trace-prompt", default=None)
    export_viz_p.add_argument(
        "--trace-output",
        default="viz/web/public/trace.json",
    )

    args = parser.parse_args(argv)

    if args.command == "generate":
        from nn_aivie.cli.generate_cmd import run_generate

        run_generate(
            args.prompt, args.checkpoint, args.max_len,
            trace_log=args.trace_log, mode=args.mode,
        )
    elif args.command == "train":
        from nn_aivie.cli.train import run_train

        run_train(
            grade=args.grade,
            phase=args.phase,
            checkpoint=args.checkpoint,
            output=args.output,
            eta=args.eta if args.eta is not None else (0.02 if args.phase == "G3" else 0.01),
            max_records=args.max_records,
            dataset=args.datasets,
            manifest=args.manifest,
            sentence_bind=not args.no_sentence_bind,
            sentence_eta_scale=args.sentence_eta_scale,
            alpha=args.alpha,
            beta=args.beta,
            teach_eos=not args.no_teach_eos,
        )
    elif args.command == "export-viz":
        from pathlib import Path

        import json

        from nn_aivie.core.state import GraphState
        from nn_aivie.viz.export_graph import export_graph_json, write_graph_json
        trace_prompt = None  # TK_019: trace v2 đã xóa

        state = GraphState.load(args.checkpoint)
        graph = export_graph_json(
            state,
            args.rules,
            min_weight=args.min_weight,
            max_edges=args.max_edges,
            hops=args.hops,
            checkpoint=str(args.checkpoint),
        )
        out = write_graph_json(args.output, graph)
        print(
            f"export-viz: {out} "
            f"({graph['meta']['n_nodes']} nodes, {graph['meta']['n_edges']} edges)"
        )
        if args.trace_prompt:
            trace = None  # TK_019: trace v2 đã xóa
            tp = Path(args.trace_output)
            tp.parent.mkdir(parents=True, exist_ok=True)
            tp.write_text(
                json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"trace: {tp} ({len(trace['steps'])} steps)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
