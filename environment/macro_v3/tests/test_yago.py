"""Real RDF streaming-reader tests with tiny locally constructed Turtle and ZIP."""
import tempfile
from pathlib import Path
import unittest
import zipfile

from environment.macro_v3.yago import load_yago

try:
    import pyoxigraph  # noqa: F401
    HAS_RDF = True
except ImportError:
    HAS_RDF = False

TURTLE = """@prefix ex: <http://example.org/> .
@prefix schema: <http://schema.org/> .
ex:Alice schema:birthPlace ex:Town .
ex:Town schema:containedInPlace ex:Country .
ex:Bob schema:memberOf ex:Team .
"""

@unittest.skipUnless(HAS_RDF, "Install pyoxigraph to run RDF reader tests")
class YagoReaderTests(unittest.TestCase):
    def test_turtle(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "yago-4.6-facts.ttl"
            path.write_text(TURTLE, encoding="utf-8")
            graph = load_yago(Path(d), max_triples=100, max_edges=20)
            self.assertEqual(len(graph.edges), 3)
            self.assertEqual(len(graph.choose_walk(__import__("random").Random(4),
                                                    min_hops=2, max_hops=2)), 2)

    def test_yago_zip(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "yago-4.6-facts.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("yago-facts.ttl", TURTLE)
            graph = load_yago(Path(d), max_triples=100, max_edges=20)
            self.assertEqual(len(graph.edges), 3)
            self.assertTrue(graph.input_file.endswith(".zip"))

if __name__ == "__main__":
    unittest.main()
