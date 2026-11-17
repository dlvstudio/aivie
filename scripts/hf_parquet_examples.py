"""Ví dụ parquet: local nn-aivie vs Hugging Face."""

import io
import textwrap

import pyarrow.parquet as pq
import requests
from datasets import load_dataset


def main() -> None:
    print("=" * 60)
    print("VÍ DỤ 1: nn-aivie local (grade_03/copula)")
    print("=" * 60)
    t = pq.read_table("data/grade_03/copula/shard_00000.parquet")
    print("Schema:", t.schema)
    print(t.to_pandas().to_string(index=False))

    print("\n" + "=" * 60)
    print("VÍ DỤ 2: HF chuẩn — stanfordnlp/imdb (text + label)")
    print("=" * 60)
    url = (
        "https://huggingface.co/datasets/stanfordnlp/imdb/"
        "resolve/refs%2Fconvert%2Fparquet/plain_text/train/0000.parquet"
    )
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    pf = pq.ParquetFile(io.BytesIO(r.content))
    print("Schema:", pf.schema)
    df = pf.read().slice(0, 2).to_pandas()
    for i, row in df.iterrows():
        print(f"\n--- row {i} ---")
        print("label:", row["label"])
        print("text:", str(row["text"])[:200] + "...")

    print("\n" + "=" * 60)
    print("VÍ DỤ 3: HF streaming — amazon_polarity (title + content + label)")
    print("=" * 60)
    ds = load_dataset("fancyzhx/amazon_polarity", split="train", streaming=True)
    for i, row in enumerate(ds):
        if i >= 2:
            break
        print(
            {
                "label": row["label"],
                "title": row["title"][:60] + "...",
                "content": row["content"][:80] + "...",
            }
        )

    print("\n" + "=" * 60)
    print("VÍ DỤ 4: nanochat-style — climbmix (repo layout + schema)")
    print("=" * 60)
    print(
        textwrap.dedent(
            """
    Repo: https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle
    Files: shard_00000.parquet ... shard_06542.parquet (~92 MB mỗi file)
    Schema (chỉ 1 cột):
      text: string

    Một dòng = một document văn bản (giống nn-aivie nhưng không có grade/domain).
    nanochat đọc: rg.column('text').to_pylist() theo row group.
            """
        ).strip()
    )
    head = requests.head(
        "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main/shard_00000.parquet",
        timeout=15,
    )
    print("shard_00000.parquet size:", head.headers.get("content-length"), "bytes")


if __name__ == "__main__":
    main()
