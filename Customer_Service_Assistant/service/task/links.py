"""Flow step link types — model how steps connect to their successors.

A link is the *next* pointer on a flow step.  It can be:

* **static** — always go to a fixed target step
* **conditional** — go to *target* when *condition* evaluates truthy
* **fallback** — go to *target* when none of the preceding conditions matched
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class FlowStepLink:
    """Base class for a next-step link.

    Attributes:
        target: The ``id`` of the step to transition to.
    """

    target: str


@dataclass(slots=True)
class StaticLink(FlowStepLink):
    """An unconditional next-step link.

    Example YAML:

    .. code-block:: yaml

        next: ask_order_number
    """

    pass


@dataclass(slots=True)
class ConditionalLink(FlowStepLink):
    """A conditional next-step link — only taken when *condition* is truthy.

    Example YAML:

    .. code-block:: yaml

        next:
          - if: "slots.get('product_id')"
            then: respond
    """

    condition: str


@dataclass(slots=True)
class FallbackLink(FlowStepLink):
    """A fallback (else) next-step link — taken when no earlier condition matched.

    Example YAML:

    .. code-block:: yaml

        next:
          - if: "slots.get('product_id')"
            then: respond
          - else: missing_product_context
    """

    pass