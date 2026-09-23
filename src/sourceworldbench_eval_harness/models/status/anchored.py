"""Anchored test-status models. Behaviour is `models.status.multi_test`,
`all_pass` included since `answers_for` keys every question off the schema."""

from abc import ABC

from sourceworldbench_eval_harness.models.status import multi_test
from sourceworldbench_eval_harness.models.status.base import SubmissionReader
from sourceworldbench_eval_harness.schemas.status import TEST_STATUS_ANCHORED


class TestStatusAnchoredModel(multi_test.MultiTestModel, ABC):
    task = "test_status_anchored"
    schema = TEST_STATUS_ANCHORED


class AllPassed(multi_test.AllPassed, TestStatusAnchoredModel):
    """The majority-class floor, and here also the "this patch broke nothing" answer.
    On `all_pass` it says nothing broke for every instance, so it never spots a
    regression and `all_pass_not_passed_f1` is zero by construction."""


class RandomStatus(multi_test.RandomStatus, TestStatusAnchoredModel):
    """Below `all_passed` on macro F1 at this imbalance. Drawing a status per test
    essentially never yields an all-PASSED instance, so its `all_pass` is the mirror
    failure — always "broken" — and `all_pass_passed_f1` is zero by construction."""


class Oracle(multi_test.Oracle, TestStatusAnchoredModel):
    pass


class FromFile(SubmissionReader, TestStatusAnchoredModel):
    pass
