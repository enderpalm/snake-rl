"""Format MCTS root stats as plain text lines (rendered in Pygame sidebar; no Tk — SDL/Tk clash on macOS)."""

from __future__ import annotations

from agents.mcts.tree import GAMMA, TreeNode, uct_score
from core.env.types import Action

_ACTION_ORDER = (Action.LEFT, Action.STRAIGHT, Action.RIGHT)
_ACTION_LABEL = {
    Action.LEFT: "LEFT",
    Action.STRAIGHT: "STRAIGHT",
    Action.RIGHT: "RIGHT",
}


def mcts_panel_lines(root: TreeNode, chosen: Action | None) -> list[str]:
    """Multiline text for Pygame: root visits, table of children, chosen action."""
    lines: list[str] = []
    cv = chosen.name if chosen is not None else "—"
    lines.append(f"MCTS root visits={root.visits}  chosen={cv}")
    lines.append("action          v  meanQ    sumQ    UCT")

    pv = root.visits
    if not root.children:
        lines.append("— (no children yet)")
        return lines

    for a in _ACTION_ORDER:
        ch = root.children.get(a)
        nm = f"{_ACTION_LABEL[a]:<8}"
        if ch is None:
            line = f"{nm}       —    —       —       —"
        else:
            v, qsum = ch.visits, ch.value
            mean = (ch.step_reward + GAMMA * qsum / v) if v else 0.0
            u = uct_score(ch, pv) if v else float("inf")
            u_s = "inf" if u == float("inf") else f"{u:.2f}"
            line = f"{nm} {v:4d} {mean:7.3f} {qsum:8.2f} {u_s:>7}"
        if chosen is not None and a == chosen:
            line = "> " + line
        else:
            line = "  " + line
        lines.append(line)
    return lines
