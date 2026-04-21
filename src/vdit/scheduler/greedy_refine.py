"""
Greedy motion-aware refinement scheduler (heap + lazy deletion).

Use case:

- You have an interpolator Interp(I_left, I_right) -> I_mid

- You have a scoring function Score(I_left, I_right) -> scalar (higher = more urgent to refine)

- Starting from an initial ordered frame sequence, keep inserting mid-frames at the

  most urgent interval until reaching target length.

This file is model-agnostic:

- Works with HTTP-based EDEN (client.py) and local EDEN (inference.py)

- Does NOT assume fixed "insert 1 or insert 3"; it inserts one frame per iteration.

Implementation notes:

- Uses a doubly-linked list of Node objects to represent current frame order.

- Uses a max-heap of intervals keyed by score.

- Uses lazy deletion: intervals in heap may become stale after insertions; we check

  adjacency + pointer versions before using.
"""

from __future__ import annotations

from dataclasses import dataclass
import os

import heapq

from typing import Any, Callable, List, Optional, Tuple


@dataclass
class Node:
    """A node in a doubly linked list holding a frame (any python object)."""

    idx: int
    frame: Any
    prev: Optional["Node"] = None
    next: Optional["Node"] = None
    # Versions for lazy deletion checks. Increment when pointers change.
    ver_next: int = 0
    ver_prev: int = 0


class _IdGen:
    def __init__(self, start: int = 0):
        self._cur = start

    def next(self) -> int:
        self._cur += 1
        return self._cur


def _link(a: Optional[Node], b: Optional[Node]) -> None:
    """Set a.next=b and b.prev=a, updating versions."""
    if a is not None:
        a.next = b
        a.ver_next += 1
    if b is not None:
        b.prev = a
        b.ver_prev += 1


def _to_list(head: Node) -> List[Any]:
    out: List[Any] = []
    cur: Optional[Node] = head
    while cur is not None:
        out.append(cur.frame)
        cur = cur.next
    return out


def _get_node_position(head: Node, target: Node) -> int:
    """计算target节点在链表中的位置（从0开始）。"""
    pos = 0
    cur: Optional[Node] = head
    while cur is not None:
        if cur is target:
            return pos
        pos += 1
        cur = cur.next
    return -1  # 未找到


def greedy_refine(
    frames: List[Any],
    target_len: int,
    score_fn: Callable[[Any, Any], float],
    interp_fn: Callable[[Any, Any], Any],
    *,
    max_iters: Optional[int] = None,
    verbose: bool = False,
    log_file: Optional[str] = None,
) -> List[Any]:
    """Greedy refinement until reaching target_len.

    Args:
        frames: initial ordered frames. Each element can be a torch Tensor, PIL.Image, etc.
                (Recommended here: each frame is a torch tensor [1,3,H,W] in [0,1].)
        target_len: desired final number of frames.
        score_fn: compute urgency score for interval (left,right). Higher -> refine earlier.
        interp_fn: generate midpoint frame for interval (left,right).
        max_iters: optional safety cap on number of insertions; default = target_len - len(frames).
        verbose: print progress.
        log_file: optional path to log file. If provided, logs every 5 insertions with frame positions and scores.

    Returns:
        A list of frames. Typically length == target_len (unless max_iters stops early).

    Raises:
        ValueError if frames is empty or target_len < len(frames).
    """
    if len(frames) == 0:
        raise ValueError("frames must be non-empty")
    if target_len < len(frames):
        raise ValueError(f"target_len ({target_len}) < initial len(frames) ({len(frames)})")
    if target_len == len(frames):
        return frames

    # Build linked list
    idgen = _IdGen(start=0)
    nodes: List[Node] = [Node(idx=idgen.next(), frame=f) for f in frames]
    for i in range(len(nodes) - 1):
        _link(nodes[i], nodes[i + 1])

    head = nodes[0]

    # Heap items:
    # (-score, tie, left_node, right_node, left_ver_next_snapshot, right_ver_prev_snapshot)
    heap: List[Tuple[float, int, Node, Node, int, int]] = []
    tie = 0

    def push_interval(left: Node, right: Node) -> None:
        nonlocal tie
        s = float(score_fn(left.frame, right.frame))
        heapq.heappush(heap, (-s, tie, left, right, left.ver_next, right.ver_prev))
        tie += 1

    # Initialize heap with all adjacent intervals
    cur = head
    while cur.next is not None:
        push_interval(cur, cur.next)
        cur = cur.next

    inserts_needed = target_len - len(frames)
    if max_iters is None:
        max_iters = inserts_needed

    inserts_done = 0
    stale_pops = 0
    log_entries: List[Tuple[int, int, int, float]] = []  # (insertion_num, left_pos, right_pos, score)

    # Open log file if specified
    log_fp = None
    if log_file:
        # Ensure the directory exists before opening the file
        log_dir = os.path.dirname(log_file)
        if log_dir:  # Only create directory if path is not empty
            os.makedirs(log_dir, exist_ok=True)
        log_fp = open(log_file, 'w', encoding='utf-8')
        log_fp.write("Greedy Refinement Log\n")
        log_fp.write("=" * 60 + "\n")
        log_fp.write(f"Initial frames: {len(frames)}\n")
        log_fp.write(f"Target frames: {target_len}\n")
        log_fp.write(f"Total insertions needed: {inserts_needed}\n")
        log_fp.write("=" * 60 + "\n")
        log_fp.write(f"{'Insertion':<12} {'Left_Pos':<10} {'Right_Pos':<11} {'Score':<10} {'Current_Total':<15}\n")
        log_fp.write("-" * 60 + "\n")

    while inserts_done < inserts_needed:
        if inserts_done >= max_iters:
            break
        if not heap:
            break

        neg_s, _, left, right, left_ver, right_ver = heapq.heappop(heap)

        # Lazy deletion check: still adjacent and versions unchanged
        if (
            left.next is not right
            or left.ver_next != left_ver
            or right.prev is not left
            or right.ver_prev != right_ver
        ):
            stale_pops += 1
            continue

        # Calculate positions for logging (before insertion)
        left_pos = _get_node_position(head, left)
        right_pos = _get_node_position(head, right)
        score = -neg_s  # Convert back from negated heap score

        # Generate midpoint
        mid_frame = interp_fn(left.frame, right.frame)
        mid_node = Node(idx=idgen.next(), frame=mid_frame)

        # Insert: left <-> mid <-> right
        _link(left, mid_node)
        _link(mid_node, right)

        inserts_done += 1

        # Update only local intervals
        push_interval(left, mid_node)
        push_interval(mid_node, right)

        # Log every 5 insertions or last one
        should_log = (inserts_done % 5 == 0 or inserts_done == inserts_needed)
        if should_log:
            cur_len = len(frames) + inserts_done
            best_s = -heap[0][0] if heap else float("nan")
            print(
                f"[greedy_refine] inserted {inserts_done}/{inserts_needed} | len={cur_len} | "
                f"best_score={best_s:.4f} | stale_pops={stale_pops}"
            )
            
            # Write to log file
            if log_fp:
                log_fp.write(f"{inserts_done:<12} {left_pos:<10} {right_pos:<11} {score:<10.6f} {cur_len:<15}\n")
                log_fp.flush()

    # Close log file
    if log_fp:
        log_fp.write("-" * 60 + "\n")
        log_fp.write(f"Total insertions: {inserts_done}\n")
        log_fp.write(f"Final frames: {len(frames) + inserts_done}\n")
        log_fp.close()

    return _to_list(head)

