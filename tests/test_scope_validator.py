"""Behavioural tests for the deterministic supported-scope gate.

Every supported topic is exercised with the wording employees actually
use -- formal, misspelled, and slang -- and every unsupported topic is
paired with the supported topic it would otherwise be answered from, so
a widened alias catalog cannot silently start answering questions this
corpus has no policy for. All tests run offline (AGENTS.md section 9).
"""

from __future__ import annotations

import unittest

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
        self.assertEqual(decision.reason, "unsupported_topic")

    def test_out_of_domain_question_is_not_supported(self) -> None:
        decision = validate_scope(OUT_OF_DOMAIN_QUERY)

        self.assertFalse(decision.supported)
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
