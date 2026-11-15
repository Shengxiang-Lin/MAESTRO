"""End-to-end small RDF-fixture smoke test for v3 CLI and exports."""
import json
import tempfile
from pathlib import Path
import unittest

from environment.macro_v3.__main__ import main

try:
    import pyoxigraph  # noqa: F401
    HAS_OXI = True
except ImportError:
    HAS_OXI = False


@unittest.skipUnless(HAS_OXI, "pyoxigraph is required for Turtle ingestion")
class V3CliTests(unittest.TestCase):
    def test_cli_produces_valid_draft_scenes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / "yago-4.6-facts.ttl"
            source.write_text(
                """@prefix x: <https://example.org/> .
@prefix s: <http://schema.org/> .
x:A s:birthPlace x:B .
x:B s:containedInPlace x:C .
x:D s:memberOf x:E .
x:E s:location x:F .
x:G s:parent x:H .
""", encoding="utf-8",
            )
            out = root / "out"
            result = main([
                "--yago", str(source), "--output-dir", str(out),
                "--target", "3", "--min-hops", "1", "--max-hops", "2",
                "--max-triples", "30", "--max-edges", "30",
            ])
            self.assertEqual(result, 0)
            with (out / "drafts" / "scenes.jsonl").open(encoding="utf-8") as f:
                scenes = [json.loads(line) for line in f]
            self.assertEqual(len(scenes), 3)
            self.assertEqual(
                json.loads((out / "summary.json").read_text())["status"],
                "pilot_done",
            )
            self.assertTrue(all(s["quality_report"]["passed"] for s in scenes))
            self.assertTrue(all(s["gold_answer"] for s in scenes))


if __name__ == "__main__":
    unittest.main()
