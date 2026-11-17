"""Natural grounded focus noun phrases for known YAGO motif chains.

The terminal focus is generated deterministically from the frozen relation
recipe; it cannot introduce an unseen fact or relax a required hop.
"""
from __future__ import annotations


_FOCUS = {
    ("birthplace", "containedinplace"):
        "the administrative area containing the identified person's birthplace",
    ("birthplace", "location"):
        "the location recorded for the identified person's birthplace",
    ("deathplace", "location"):
        "the location recorded for the identified person's place of death",
    ("doctoraladvisor", "fieldofwork"):
        "the research field of the identified scholar's doctoral advisor",
    ("doctoraladvisor", "birthplace"):
        "the birthplace of the identified scholar's doctoral advisor",
    ("spouse", "birthplace"):
        "the birthplace of the identified person's spouse",
    ("memberof", "location"):
        "the location of the organization affiliated with the identified entity",
    ("location", "containedinplace"):
        "the administrative area containing the identified entity's location",
    ("containedinplace", "containedinplace"):
        "the larger administrative area containing the region of the identified place",
    ("neighbor", "location"):
        "the location of a neighboring place of the identified place",
    ("location", "neighbor"):
        "the place neighboring the identified entity's location",
    ("birthplace", "containedinplace", "containedinplace"):
        "the larger administrative area containing the region of the identified person's birthplace",
    ("birthplace", "location", "containedinplace"):
        "the administrative area containing the recorded location of the identified person's birthplace",
    ("doctoraladvisor", "birthplace", "location"):
        "the location recorded for the birthplace of the identified scholar's doctoral advisor",
    ("doctoraladvisor", "birthplace", "containedinplace"):
        "the administrative area containing the birthplace of the identified scholar's doctoral advisor",
    ("memberof", "location", "containedinplace"):
        "the administrative area containing the location of the identified entity's affiliated organization",
    ("location", "containedinplace", "containedinplace"):
        "the larger administrative area containing the region of the identified entity's location",
    ("spouse", "birthplace", "containedinplace"):
        "the administrative area containing the birthplace of the identified person's spouse",
    ("deathplace", "location", "containedinplace"):
        "the administrative area containing the recorded location of the identified person's place of death",
    ("nationality", "capital", "location"):
        "the location of the capital of the identified person's nationality",
    ("location", "neighbor", "location"):
        "the location recorded for the place neighboring the identified entity's location",
    ("deathplace", "location", "neighbor"):
        "the area neighboring the location of the identified person's place of death",
    ("location", "capital", "twinnedwith"):
        "the municipality twinned with the capital of the identified entity's location",
}


def focus_phrase(keys: tuple[str, ...]) -> str:
    try:
        return _FOCUS[keys]
    except KeyError as exc:
        # Fail closed, rather than silently dropping a semantic relation.
        raise ValueError("unsupported_natural_goal_focus:" + repr(keys)) from exc


def focus_coverage() -> frozenset[tuple[str, ...]]:
    return frozenset(_FOCUS)
