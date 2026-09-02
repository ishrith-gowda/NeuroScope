"""unit tests for the generality-benchmark modules (GTA5->Cityscapes fidelity-vs-utility).

these guard the result-assembly + model-reconstruction logic that produces the paper's headline
table/figure: a silent bug in tag parsing, dmIoU, or the CycleGAN generator would corrupt every
reported number. mirrors tests/test_metrics.py: the generality scripts are not a package, so we add
their directory to sys.path and import them directly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

GEN = Path(__file__).resolve().parents[1] / "journal_extension" / "generality"
sys.path.insert(0, str(GEN))

import aggregate_generality as agg  # noqa: E402
import plot_frontier as pf  # noqa: E402


@pytest.mark.parametrize(
    "tag,expected",
    [
        ("sdedit_s055", 0.55),
        ("sdedit_s030", 0.30),
        ("sdedit_s070", 0.70),
        ("sdedit_s055_empty", 0.55),
        ("raw", None),
        ("cyclegan", None),
        ("colormatch", None),
    ],
)
def test_strength_of(tag, expected):
    assert pf.strength_of(tag) == expected


@pytest.mark.parametrize(
    "tag,family",
    [
        ("raw", "baseline"),
        ("colormatch", "non-learned"),
        ("cyclegan", "learned-GAN"),
        ("sdedit_s055", "learned-diffusion"),
        ("sdedit_s040", "learned-diffusion"),
        ("sdedit_s055_empty", "learned-diffusion"),
    ],
)
def test_family_classification(tag, family):
    fam, label = agg._family(tag)
    assert fam == family
    assert isinstance(label, str) and label


def test_family_diffusion_label_encodes_strength_and_prompt():
    _, label = agg._family("sdedit_s055")
    assert "0.55" in label
    _, elabel = agg._family("sdedit_s055_empty")
    assert "0.55" in elabel and "empty" in elabel.lower()


def _write(d: Path, tag: str, miou: float, fid: float | None) -> None:
    (d / f"{tag}.json").write_text(
        json.dumps({"tag": tag, "mIoU": miou, "FID": fid, "n_images": 200})
    )


def test_aggregate_end_to_end(tmp_path, monkeypatch):
    # synthetic result set spanning all families
    _write(tmp_path, "raw", 0.3475, 165.79)
    _write(tmp_path, "colormatch", 0.3185, 194.96)
    _write(tmp_path, "cyclegan", 0.1263, 105.75)
    _write(tmp_path, "sdedit_s030", 0.2529, 162.60)
    _write(tmp_path, "sdedit_s070", 0.0983, 122.02)

    monkeypatch.setattr(sys, "argv", ["aggregate", "--results", str(tmp_path)])
    agg.main()

    assert (tmp_path / "benchmark_table.md").exists()
    assert (tmp_path / "benchmark_table.tex").exists()
    combined = json.loads((tmp_path / "benchmark_combined.json").read_text())

    assert combined["raw_mIoU"] == pytest.approx(0.3475)
    rows = {r["tag"]: r for r in combined["rows"]}
    # dmIoU is measured against raw
    assert rows["cyclegan"]["dmIoU_vs_raw"] == pytest.approx(0.1263 - 0.3475)
    assert rows["raw"]["dmIoU_vs_raw"] == pytest.approx(0.0)
    # rows are sorted by FID ascending -> cyclegan (best FID) first
    assert combined["rows"][0]["tag"] == "cyclegan"
    # diffusion frontier holds only the two sdedit points
    assert {p["strength"] for p in combined["diffusion_frontier"]} == {
        agg._family("sdedit_s030")[1],
        agg._family("sdedit_s070")[1],
    }


def test_cyclegan_generator_shape_and_range():
    torch = pytest.importorskip("torch")
    import cyclegan_translate as cg

    net = cg.ResnetGenerator().eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        y = net(x)
    assert y.shape == (1, 3, 64, 64)
    # final layer is Tanh -> outputs bounded in [-1, 1]
    assert float(y.min()) >= -1.0001 and float(y.max()) <= 1.0001
