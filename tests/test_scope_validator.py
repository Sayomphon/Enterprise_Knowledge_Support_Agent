"""Behavioural tests for the deterministic supported-scope gate.

Every supported topic is exercised with the wording employees actually
use -- formal, misspelled, and slang -- and every unsupported topic is
paired with the supported topic it would otherwise be answered from, so
a widened alias catalog cannot silently start answering questions this
corpus has no policy for. All tests run offline (AGENTS.md section 9).
"""

from __future__ import annotations

import unittest

from src.query_expansion import alias_expansion_variants
from src.guardrails.scope_validator import (
    SUPPORTED_TOPIC_ALIASES,
    UNSUPPORTED_TOPIC_ALIASES,
    validate_scope,
)
from src.schemas import KNOWLEDGE_TOPICS

# (label, query, topic that must be resolved) for supported wording.
SUPPORTED_QUERIES = (
    ("annual_leave_exact", "พนักงานมีสิทธิ์ลาพักร้อนกี่วันต่อปี", "annual_leave"),
    ("annual_leave_typo", "ลาพักรอ้น 2 วันกดตรงไหนอะ", "annual_leave"),
    ("annual_leave_synonym", "ลาพักผ่อนต้องแจ้งล่วงหน้าไหม", "annual_leave"),
    ("sick_leave_exact", "ลาป่วยต้องแจ้งหัวหน้าภายในกี่โมง", "sick_leave"),
    ("receipt_slang", "ใบเสดหาย เคลมได้มั้ย", "receipt_policy"),
    ("receipt_exact", "ใบเสร็จอิเล็กทรอนิกส์ใช้ได้ไหม", "receipt_policy"),
    (
        "reimbursement_slang",
        "เบิกตังค่า taxi ได้ปะ",
        "reimbursement_process",
    ),
    ("wfh_thai", "ทำงานจากที่บ้านได้สัปดาห์ละกี่วัน", "work_from_home"),
    ("wfh_english", "How many WFH days per week do I get?", "work_from_home"),
)

# (label, query, supported topic it must NOT be answered from).
UNSUPPORTED_QUERIES = (
    ("maternity_leave", "ลาคลอดต้องใช้ใบรับรองแพทย์ไหม", "sick_leave"),
    ("marriage_leave", "ลาแต่งงานใช้สิทธิ์ได้กี่วัน", "annual_leave"),
    ("ordination_leave", "ลาบวชต้องแจ้งล่วงหน้ากี่วัน", "annual_leave"),
    (
        "medical_reimbursement",
        "เบิกค่ารักษาพยาบาลกับบริษัทได้ไหม",
        "reimbursement_process",
    ),
    (
        "meal_reimbursement",
        "ค่าอาหารระหว่างทำงานเบิกได้ไหม",
        "reimbursement_process",
    ),
    ("payroll_date", "เงินเดือนออกวันไหน", "reimbursement_process"),
    (
        "salary",
        "เงินเดือนของหัวหน้าฝ่ายการเงินคือเท่าไหร่",
        "reimbursement_process",
    ),
)

# Eligibility questions about expense items this corpus has no rule for,
# each paired with a supported question one word away from it. The
# reimbursement PROCESS policy describes how to file a claim and never
# says which items qualify, so its wording used to carry these past the
# gate and buy them a confident yes. The benign half is the control: an
# alias that raises the block rate by refusing real questions is a
# regression, not a fix.
NEAR_DOMAIN_PAIRS = (
    (
        "phone",
        "ขั้นตอนยื่นเบิกค่าโทรศัพท์มือถือผ่าน Expense Portal",
        "ขั้นตอนการเบิกค่าแท็กซี่หลังทำ OT ต้องทำอย่างไร",
        "reimbursement_process",
    ),
    (
        "internet",
        "ขอเบิกค่าอินเทอร์เน็ตทำงานจากที่บ้านผ่าน Expense Portal ได้ไหม",
        "ทำงานจากที่บ้านได้สัปดาห์ละกี่วัน",
        "work_from_home",
    ),
    (
        "office_equipment",
        "ค่าอุปกรณ์สำนักงานเบิกได้ไหม",
        "ใบเสร็จหายต้องทำอย่างไรถึงจะเบิกได้",
        "receipt_policy",
    ),
    (
        "training_expense",
        "ขอเบิกค่าอบรมภายนอกต้องยื่นเอกสารอะไรบ้าง",
        "เบิกค่าที่จอดรถตอนไปหาลูกค้าได้ป่าว",
        "reimbursement_process",
    ),
    (
        "per_diem",
        "เบี้ยเลี้ยงเดินทางต่างจังหวัดได้วันละเท่าไหร่",
        "ค่าเดินทางไปพบลูกค้าต่างจังหวัดเบิกยังไง",
        "reimbursement_process",
    ),
    (
        "business_leave",
        "ลากิจใช้สิทธิ์ได้กี่วันต่อปี",
        "ลาพักผ่อนใช้สิทธิ์ได้กี่วันต่อปี",
        "annual_leave",
    ),
    (
        "fuel_mileage",
        "ขับรถตัวเองไปพบลูกค้าเบิกค่าน้ำมันได้ไหม",
        "ขับแท็กซี่กลับบ้านหลัง OT เบิกค่าแท็กซี่ได้ไหม",
        "reimbursement_process",
    ),
)

UNDER_SPECIFIED_QUERY = "ลาได้กี่วัน"
OUT_OF_DOMAIN_QUERY = "Bitcoin วันนี้ราคาเท่าไหร่"


class TestSupportedTopics(unittest.TestCase):
    """Wording employees really use must resolve to the right topic."""

    def test_supported_queries_resolve_their_topic(self) -> None:
        for label, query, expected_topic in SUPPORTED_QUERIES:
            with self.subTest(case=label):
                decision = validate_scope(query)

                self.assertTrue(decision.supported)
                self.assertIn(expected_topic, decision.topics)
                self.assertIsNone(decision.reason)

    def test_supported_decision_reports_its_alias_score(self) -> None:
        decision = validate_scope("ลาป่วยต้องแจ้งหัวหน้าภายในกี่โมง")

        self.assertGreaterEqual(decision.score, 0.0)
        self.assertLessEqual(decision.score, 1.0)


class TestUnsupportedTopics(unittest.TestCase):
    """In-domain questions without a policy must be refused explicitly."""

    def test_unsupported_queries_never_resolve_a_supported_topic(
        self,
    ) -> None:
        for label, query, forbidden_topic in UNSUPPORTED_QUERIES:
            with self.subTest(case=label):
                decision = validate_scope(query)

                self.assertFalse(decision.supported)
                self.assertEqual(decision.reason, "unsupported_topic")
                self.assertNotIn(forbidden_topic, decision.topics)
                self.assertEqual(decision.topics, ())

    def test_under_specified_leave_question_is_not_supported(self) -> None:
        # "How many days of leave" names no leave type, so no policy can
        # answer it without guessing which one the employee meant.
        decision = validate_scope(UNDER_SPECIFIED_QUERY)

        self.assertFalse(decision.supported)
        self.assertEqual(decision.reason, "ambiguous_topic")

    def test_out_of_domain_question_is_not_supported(self) -> None:
        decision = validate_scope(OUT_OF_DOMAIN_QUERY)

        self.assertFalse(decision.supported)
        self.assertEqual(decision.reason, "unsupported_topic")


class TestAmbiguousTopics(unittest.TestCase):
    """Under-specified questions are separated from unanswerable ones.

    Both verdicts refuse the request, so nothing here is about routing:
    it is about which of two sentences the employee reads, and asking an
    out-of-domain asker to name their leave type would be the worse of
    the two mistakes. The tests are therefore paired -- one query that
    must earn the new code, and the neighbours that must not.
    """

    def test_a_question_touching_two_topics_is_reported_as_ambiguous(
        self,
    ) -> None:
        # "ลา" is shared by the annual-leave and sick-leave aliases, so
        # this query brushes both and resolves neither.
        decision = validate_scope(UNDER_SPECIFIED_QUERY)

        self.assertEqual(decision.reason, "ambiguous_topic")
        # A near match is not a resolved topic: publishing one here
        # would put a refused topic into the alias expansion and the
        # telemetry alike.
        self.assertEqual(decision.topics, ())

    def test_out_of_domain_questions_keep_the_unsupported_code(
        self,
    ) -> None:
        # The benign-lookalike half of the pair. Each of these brushes at
        # most one topic by accident, which is what the second-topic rule
        # exists to distinguish from an unfinished question.
        for label, query in (
            ("crypto price", OUT_OF_DOMAIN_QUERY),
            ("football", "แมนยูคืนนี้เตะกี่โมง"),
            ("recipe", "วิธีทำต้มยำกุ้งให้อร่อย"),
            ("laptop", "how do I reset my laptop"),
        ):
            with self.subTest(case=label):
                decision = validate_scope(query)

                self.assertEqual(decision.reason, "unsupported_topic")

    def test_a_named_unsupported_topic_is_never_called_ambiguous(
        self,
    ) -> None:
        # The corpus has no ordination-leave policy, and the question is
        # perfectly specific. Asking this employee to be more precise
        # would be a worse answer than admitting the gap.
        decision = validate_scope("ลาบวชต้องแจ้งล่วงหน้ากี่วัน")

        self.assertEqual(decision.reason, "unsupported_topic")

    def test_a_resolved_topic_is_never_called_ambiguous(self) -> None:
        # The same question with the leave type named. It must answer,
        # not ask again.
        decision = validate_scope("ลาพักร้อนได้กี่วัน")

        self.assertTrue(decision.supported)
        self.assertIsNone(decision.reason)

    def test_raising_the_floor_above_the_second_topic_removes_the_verdict(
        self,
    ) -> None:
        # Pins the rule to the score rather than to the wording: with the
        # floor above the second topic's score the same query is a
        # single accidental match again.
        decision = validate_scope(
            UNDER_SPECIFIED_QUERY, ambiguous_min_score=0.14
        )

        self.assertEqual(decision.reason, "unsupported_topic")


class TestNearDomainExpenseItems(unittest.TestCase):
    """Item eligibility is refused; the process questions still answer.

    Both halves are asserted together on purpose. The catalog is the only
    thing separating "how do I claim a taxi" from "can I claim my phone
    bill", and the two are lexically almost identical, so an alias wide
    enough to catch the second will catch the first if nobody checks.
    """

    def test_unsupported_expense_items_are_refused(self) -> None:
        for label, attack, _benign, _topic in NEAR_DOMAIN_PAIRS:
            with self.subTest(case=label):
                decision = validate_scope(attack)

                self.assertFalse(decision.supported)
                self.assertEqual(decision.reason, "unsupported_topic")
                self.assertEqual(decision.topics, ())

    def test_the_benign_twin_still_resolves_its_topic(self) -> None:
        for label, _attack, benign, topic in NEAR_DOMAIN_PAIRS:
            with self.subTest(case=label):
                decision = validate_scope(benign)

                self.assertTrue(decision.supported)
                self.assertIn(topic, decision.topics)

    def test_fuel_aliases_do_not_swallow_the_travel_wording(self) -> None:
        # "kha-doen-thang" (travel) is a SUPPORTED alias: taxi fares after
        # overtime are covered. A fuel alias widened towards it would
        # refuse them, so the narrow spelling is load-bearing.
        decision = validate_scope("ค่าเดินทางหลังทำ OT เบิกได้ไหม")

        self.assertTrue(decision.supported)
        self.assertIn("reimbursement_process", decision.topics)


class TestReceiptEvidenceAliases(unittest.TestCase):
    """The document's own wording for evidence resolves to its topic.

    ``FIN-002`` opens by requiring "financial evidence" and naming the
    "documents supporting a claim"; a paraphrased receipt question that
    never says the word "receipt" used to resolve to nothing but the
    process topic, and the deterministic expansion then had no receipt
    wording to search with. The pair below is the control: the addition
    must not pull an unrelated document question into the topic.
    """

    def test_evidence_wording_resolves_the_receipt_topic(self) -> None:
        for query in (
            "หลักฐานทางการเงินที่ใช้เบิกได้มีอะไรบ้าง",
            "ต้องแนบเอกสารประกอบการเบิกอะไรบ้าง",
        ):
            with self.subTest(query=query):
                decision = validate_scope(query)

                self.assertTrue(decision.supported)
                self.assertIn("receipt_policy", decision.topics)

    def test_a_payroll_document_request_is_still_unsupported(self) -> None:
        # One word away and out of scope: the corpus has no payroll
        # policy, and the unsupported catalog has to keep winning the
        # tie-break against the new evidence aliases.
        decision = validate_scope("ขอเอกสารรับรองเงินเดือนได้ที่ไหน")

        self.assertFalse(decision.supported)
        self.assertEqual(decision.reason, "unsupported_topic")

    def test_the_new_aliases_reach_the_expansion_variants(self) -> None:
        # The scope decision is only half of what these aliases are for:
        # the medium band searches them, which is what lifts a
        # paraphrased receipt question without touching a threshold.
        variants = alias_expansion_variants(
            "โอนเงินผ่านแอปแล้วแคปหน้าจอมาเบิกได้มั้ย",
            ["receipt_policy"],
        )

        self.assertTrue(
            any("หลักฐานทางการเงิน" in variant for variant in variants)
        )


class TestCatalogContract(unittest.TestCase):
    """The catalogs themselves must stay consistent with the corpus."""

    def test_supported_catalog_covers_exactly_the_known_topics(self) -> None:
        self.assertEqual(
            tuple(sorted(SUPPORTED_TOPIC_ALIASES)), tuple(sorted(KNOWLEDGE_TOPICS))
        )

    def test_unsupported_catalog_names_no_supported_topic(self) -> None:
        self.assertEqual(
            set(UNSUPPORTED_TOPIC_ALIASES) & set(KNOWLEDGE_TOPICS), set()
        )

    def test_every_catalog_alias_is_a_non_empty_string(self) -> None:
        catalogs = (SUPPORTED_TOPIC_ALIASES, UNSUPPORTED_TOPIC_ALIASES)
        for catalog in catalogs:
            for topic, aliases in catalog.items():
                with self.subTest(topic=topic):
                    self.assertTrue(aliases)
                    for alias in aliases:
                        self.assertTrue(alias.strip())

    def test_threshold_is_injectable_for_calibration_sweeps(self) -> None:
        # A threshold above every possible score rejects everything, which
        # proves the sweep really controls the decision.
        decision = validate_scope(
            "ลาป่วยต้องแจ้งหัวหน้าภายในกี่โมง", match_threshold=1.01
        )

        self.assertFalse(decision.supported)


if __name__ == "__main__":
    unittest.main()
