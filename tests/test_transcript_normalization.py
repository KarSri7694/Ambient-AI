import asyncio
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from application.services.transcript_normalization_service import TranscriptNormalizationService


class TranscriptNormalizationTests(unittest.TestCase):
    def test_heuristic_normalizer_merges_fragment_into_named_speaker(self):
        service = TranscriptNormalizationService(llm_provider=None)
        transcript = (
            "[4.8066 - 9.9366] -> SPEAKER_01- [-0.472%]: College\n"
            "[10.3585 - 13.3791] -> Kartikeya- [46.853%]: fees  is  1  lakh  74000.  Last\n"
            "[13.7841 - 15.8597] -> Kartikeya- [37.812%]: date  is  30  first  July.\n"
        )

        normalized = asyncio.run(service.normalize(transcript))

        self.assertIn("-> Kartikeya:", normalized)
        self.assertNotIn("SPEAKER_01- [-0.472%]", normalized)
        self.assertNotIn("Kartikeya- [46.853%]", normalized)
        self.assertIn("College fees is 1 lakh 74000. Last date is 30 first July.", normalized)
        self.assertEqual(len([line for line in normalized.splitlines() if line.strip()]), 1)


if __name__ == "__main__":
    unittest.main()
