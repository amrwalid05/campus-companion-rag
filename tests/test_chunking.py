import unittest

from rag_engine import chunk_text, has_relevant_evidence


class ChunkingTests(unittest.TestCase):
    def test_empty_text(self):
        self.assertEqual(chunk_text(""), [])

    def test_short_text_returns_one_chunk(self):
        self.assertEqual(chunk_text("A short policy sentence."), ["A short policy sentence."])

    def test_invalid_overlap_is_rejected(self):
        with self.assertRaises(ValueError):
            chunk_text("Some text.", chunk_size=20, overlap=20)

    def test_long_text_is_split_within_limit(self):
        text = " ".join([f"Sentence number {i} explains a university policy." for i in range(40)])
        chunks = chunk_text(text, chunk_size=180, overlap=30)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 180 for chunk in chunks))

    def test_long_sentence_falls_back_to_word_boundaries(self):
        text = " ".join(["universitypolicy"] * 80)
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 100 for chunk in chunks))

    def test_relevance_guardrail(self):
        self.assertTrue(has_relevant_evidence([{"distance": 0.4}], 0.82))
        self.assertFalse(has_relevant_evidence([{"distance": 0.95}], 0.82))
        self.assertFalse(has_relevant_evidence([], 0.82))


if __name__ == "__main__":
    unittest.main()
