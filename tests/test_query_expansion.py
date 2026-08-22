"""Unit tests for deterministic alias expansion.

These fix the two properties the medium band now relies on: the variants
are a pure function of the query and its resolved topics, so the same
question always produces the same searches, and the provenance label
reports which query actually earned a document's score.
"""

import unittest

from src.guardrails.scope_validator import (
    SUPPORTED_TOPIC_ALIASES,
    validate_scope,
)
from src.query_expansion import (
    _ALIASES_PER_TOPIC,
    alias_expansion_variants,
    label_alias_matches,
)
from src.schemas import RetrievedDocument

WFH_QUERY = "ขอ wfh อาทิตย์นึงได้กี่วันอะ"
ENGLISH_ANNUAL_LEAVE_QUERY = "How many annual leave days do I get?"
# Sick-leave vocabulary, in both languages. The expansion searches the
# aliases of whatever the scope gate resolved, so a contaminated topic
# set puts these words into an annual-leave search and lifts the wrong
# policy up the ranking.
SICK_LEAVE_WORDS = ("sick leave", "ลาป่วย", "ใบรับรองแพทย์")


def _document(matched_query: str) -> RetrievedDocument:
    """Build one retrieved document that matched the given query."""
    return RetrievedDocument(
        source_id="HR-003",
        title="WFH policy",
        source_type="policy",
        content="T",
        score=0.3,
        authority="authoritative",
        status="active",
        topics=("work_from_home",),
        matched_query=matched_query,
        matched_query_type="original",
    )


class TestAliasExpansionVariants(unittest.TestCase):
    """What the expansion searches, and what it refuses to search."""

    def test_no_topic_produces_no_variant(self) -> None:
        # The low band and the unsupported branch never reach this node,
        # but an empty topic list must still mean "search nothing extra"
        # rather than "search the whole catalog".
        self.assertEqual(alias_expansion_variants(WFH_QUERY, []), [])

    def test_unknown_topic_is_ignored_rather_than_raising(self) -> None:
        # Topics arrive from pipeline state, not from a literal.
        self.assertEqual(alias_expansion_variants(WFH_QUERY, ["nonsense"]), [])

    def test_one_topic_produces_the_combined_and_alias_only_variants(
        self,
    ) -> None:
        variants = alias_expansion_variants(WFH_QUERY, ["work_from_home"])

        self.assertEqual(len(variants), 2)
        self.assertTrue(variants[0].startswith(WFH_QUERY))
        self.assertEqual(variants[1], variants[0][len(WFH_QUERY) + 1 :])

    def test_a_leave_question_never_searches_the_other_leave_type(
        self,
    ) -> None:
        variants = alias_expansion_variants(
            ENGLISH_ANNUAL_LEAVE_QUERY,
            validate_scope(ENGLISH_ANNUAL_LEAVE_QUERY).topics,
        )

        for word in SICK_LEAVE_WORDS:
            with self.subTest(word=word):
                self.assertNotIn(word, " ".join(variants))

    def test_the_best_matching_aliases_come_first(self) -> None:
        # "wfh" appears verbatim in the query, so it must lead: the point
        # of the variant is to add the corpus wording for THIS question.
        variants = alias_expansion_variants(WFH_QUERY, ["work_from_home"])

        self.assertTrue(variants[1].startswith("wfh"))

    def test_at_most_three_aliases_enter_one_variant(self) -> None:
        # reimbursement_process carries twelve aliases; pooling all of
        # them would dilute the fragments that separate the documents.
        variants = alias_expansion_variants(
            "เบิกตังค่า taxi ได้ปะ", ["reimbursement_process"]
        )

        self.assertEqual(
            len(variants[1].split(" ")),
            _ALIASES_PER_TOPIC,
        )

    def test_two_topics_produce_two_variants_each(self) -> None:
        variants = alias_expansion_variants(
            "ใบเสร็จหายต้องทำอย่างไรถึงจะเบิกได้",
            ["receipt_policy", "reimbursement_process"],
        )

        self.assertEqual(len(variants), 4)

    def test_the_same_input_always_produces_the_same_variants(self) -> None:
        # Determinism is the whole point: this branch replaces the only
        # part of the pipeline whose route depended on a provider.
        first = alias_expansion_variants(
            WFH_QUERY, ["work_from_home", "annual_leave"]
        )
        second = alias_expansion_variants(
            WFH_QUERY, ["work_from_home", "annual_leave"]
        )

        self.assertEqual(first, second)

    def test_variants_are_deduplicated(self) -> None:
        # A repeated topic must not buy a repeated column in the
        # similarity matrix; the score is max-pooled either way.
        repeated = alias_expansion_variants(
            WFH_QUERY, ["work_from_home", "work_from_home"]
        )

        self.assertEqual(len(repeated), 2)

    def test_every_alias_in_a_variant_comes_from_the_catalog(self) -> None:
        variants = alias_expansion_variants(WFH_QUERY, ["work_from_home"])

        for alias in variants[1].split(" "):
            with self.subTest(alias=alias):
                self.assertIn(
                    alias,
                    " ".join(SUPPORTED_TOPIC_ALIASES["work_from_home"]),
                )


class TestAliasProvenanceLabel(unittest.TestCase):
    """A reviewer must see when a document ranked on catalog wording."""

    def test_a_document_matched_by_a_variant_is_labelled_alias(self) -> None:
        labelled = label_alias_matches(
            [_document("wfh ทำงานที่บ้าน")], ["wfh ทำงานที่บ้าน"], WFH_QUERY
        )

        self.assertEqual(labelled[0].matched_query_type, "alias")

    def test_a_document_matched_by_the_original_query_is_left_alone(
        self,
    ) -> None:
        labelled = label_alias_matches(
            [_document(WFH_QUERY)], ["wfh ทำงานที่บ้าน"], WFH_QUERY
        )

        self.assertEqual(labelled[0].matched_query_type, "original")

    def test_the_original_query_wins_a_collision_with_a_variant(self) -> None:
        # A query that is exactly its own alias would otherwise be
        # reported as machine-generated wording.
        labelled = label_alias_matches(
            [_document(WFH_QUERY)], [WFH_QUERY], WFH_QUERY
        )

        self.assertEqual(labelled[0].matched_query_type, "original")

    def test_labelling_does_not_mutate_the_input_documents(self) -> None:
        original = _document("wfh ทำงานที่บ้าน")

        label_alias_matches([original], ["wfh ทำงานที่บ้าน"], WFH_QUERY)

        self.assertEqual(original.matched_query_type, "original")


if __name__ == "__main__":
    unittest.main()
