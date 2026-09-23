"""Status vocabularies and the questions asked about them, one entry per dataset."""

from dataclasses import dataclass

PASSED = "PASSED"
FAILED = "FAILED"
ERROR = "ERROR"

NOT_PASSED = "NOT_PASSED"
# Order matters for `metrics.multiclass_mcc`: the positive class comes last.
BINARY_LABELS: tuple[str, ...] = (PASSED, NOT_PASSED)


def to_binary(label: str) -> str:
    return PASSED if label == PASSED else NOT_PASSED


# A question is named after the dataset prompt it answers, minus the `Q-` prefix and
# with dashes as underscores, so a submitter reading `Q-outcome` writes `outcome`.
#     Q-binary   -> binary    PASSED or NOT_PASSED
#     Q-outcome  -> outcome   the full status
#     Q-all-pass -> all_pass  does every test still pass?
BINARY = "binary"
OUTCOME = "outcome"
ALL_PASS = "all_pass"


@dataclass(frozen=True)
class StatusSchema:
    """Frozen because a task installs its settings on the config that
    `write_config_beside` dumps after scoring, so a vocabulary that could be edited
    afterwards would leave a sidecar describing a run that never happened.
    """

    labels: tuple[str, ...]
    questions: tuple[str, ...]

    @property
    def label_vocabulary(self) -> dict[str, tuple[str, ...]]:
        """`{question: allowed labels}`, omitting questions whose answer is not a
        label — `all_pass` is a boolean.
        """
        by_question = {BINARY: BINARY_LABELS, OUTCOME: self.labels}
        return {question: labels for question, labels in by_question.items() if question in self.questions}


# The dataset filters SKIPPED and OTHER out; one test per row, so there is no set for
# `all_pass` to ask about.
SINGLE_TEST_STATUS = StatusSchema(
    labels=(PASSED, FAILED, ERROR),
    questions=(BINARY, OUTCOME),
)

TEST_STATUS_ANCHORED = StatusSchema(
    labels=(PASSED, FAILED, ERROR),
    questions=(BINARY, OUTCOME, ALL_PASS),
)


__all__ = [
    "ALL_PASS",
    "BINARY",
    "BINARY_LABELS",
    "ERROR",
    "FAILED",
    "NOT_PASSED",
    "OUTCOME",
    "PASSED",
    "SINGLE_TEST_STATUS",
    "TEST_STATUS_ANCHORED",
    "StatusSchema",
    "to_binary",
]
