"""Offline tests for the pulsed docking control law (robot OFF)."""

from dock_node.dock_controller import (
    DockConfig, DockController, Phase, RearRange, TagObs)


def _ctrl(**over):
    # small pulse timings so tests can step deterministically across pulse/settle
    base = dict(pulse_dur=0.10, settle_dur=0.10, align_tol=0.15,
                reverse_correct_tol=0.30, reverse_speed=0.18)
    base.update(over)
    cfg = DockConfig(**base)
    return DockController(cfg), cfg


# ── search ──────────────────────────────────────────────────────────────────
def test_search_pulses_then_fails_on_timeout():
    c, _ = _ctrl(t_search_max=5.0)
    c.start(0.0, TagObs(seen=False))
    cmd = c.update(TagObs(seen=False), RearRange(), 0.0)
    assert cmd.phase is Phase.SEARCH and cmd.angular_z != 0.0   # pulsing to sweep
    cmd = c.update(TagObs(seen=False), RearRange(), 10.0)
    assert cmd.failed


def test_search_sweeps_BOTH_directions():
    """A one-way sweep can only find a tag it is already turning toward. Regression:
    2026-07-30 OMNI rotated left, away from the tag, until the search timed out."""
    c, cfg = _ctrl(pulse_dur=0.10, settle_dur=0.10, t_search_max=999.0)
    c.start(0.0, TagObs(seen=False))
    dirs, t = [], 0.0
    for _ in range(400):
        cmd = c.update(TagObs(seen=False), RearRange(), t)
        if cmd.angular_z != 0.0:
            s = 1 if cmd.angular_z > 0 else -1
            if not dirs or dirs[-1] != s:
                dirs.append(s)
        t += 0.05
    assert len(set(dirs)) == 2, f'sweep never reversed direction: {dirs}'
    assert len(dirs) >= 3, f'expected several reversals, got {dirs}'


def test_search_legs_widen():
    """Each leg should be longer than the last so the arc grows outwards."""
    c, _ = _ctrl(t_search_max=999.0)
    c.start(0.0, TagObs(seen=False))
    assert c._search_leg_len == 2
    for _ in range(2):
        c._search_in_leg = c._search_leg_len
        c._pulse_reset()
        c.update(TagObs(seen=False), RearRange(), 1.0)
    assert c._search_leg_len > 2


def test_search_state_resets_on_reentry():
    c, _ = _ctrl(t_search_max=999.0)
    c.start(0.0, TagObs(seen=False))
    c._search_leg_len = 12
    c._search_dir = -1.0
    c._enter(Phase.SEARCH, 5.0)
    assert c._search_leg_len == 2 and c._search_dir == 1.0


def test_search_acquires_tag_to_align():
    c, _ = _ctrl()
    c.start(0.0, TagObs(seen=False))
    cmd = c.update(TagObs(seen=True, ex=0.4), RearRange(), 0.3)
    assert cmd.phase is Phase.ALIGN


def test_start_with_tag_begins_in_align():
    c, _ = _ctrl()
    c.start(0.0, TagObs(seen=True, ex=0.3))
    assert c.phase is Phase.ALIGN


# ── pulse mechanics ───────────────────────────────────────────────────────────
def test_pulse_shape_pulsing_then_settle_then_redecide():
    c, cfg = _ctrl(pulse_dur=0.10, settle_dur=0.10, align_tol=0.10)
    c.start(0.0, TagObs(seen=True, ex=0.5))
    # t=0: idle → start pulse (rotating)
    assert c.update(TagObs(seen=True, ex=0.5), RearRange(), 0.0).angular_z != 0.0
    # mid-pulse: still rotating
    assert c.update(TagObs(seen=True, ex=0.5), RearRange(), 0.05).angular_z != 0.0
    # after pulse_dur: settling → zero
    assert c.update(TagObs(seen=True, ex=0.5), RearRange(), 0.12).angular_z == 0.0
    # settle completes on this tick (still zero), then next idle tick re-decides
    c.update(TagObs(seen=True, ex=0.5), RearRange(), 0.22)
    cmd = c.update(TagObs(seen=True, ex=0.0), RearRange(), 0.30)   # centered now → reverse
    assert cmd.phase is Phase.REVERSE


# ── align ─────────────────────────────────────────────────────────────────────
def test_align_pulse_turns_toward_center():
    c, cfg = _ctrl(steer_sign=1.0)
    c.start(0.0, TagObs(seen=True, ex=0.5))
    cmd = c.update(TagObs(seen=True, ex=0.5), RearRange(), 0.0)
    assert cmd.phase is Phase.ALIGN and cmd.angular_z < 0.0   # + ex → negative yaw (steer_sign +1)


def test_align_pulse_flips_with_steer_sign():
    c1, _ = _ctrl(steer_sign=1.0)
    c1.start(0.0, TagObs(seen=True, ex=0.5))
    right = c1.update(TagObs(seen=True, ex=0.5), RearRange(), 0.0).angular_z
    c2, _ = _ctrl(steer_sign=-1.0)
    c2.start(0.0, TagObs(seen=True, ex=0.5))
    left = c2.update(TagObs(seen=True, ex=0.5), RearRange(), 0.0).angular_z
    assert right == -left and right != 0.0


def test_align_reverses_immediately_when_already_centered():
    c, _ = _ctrl(align_tol=0.15)
    c.start(0.0, TagObs(seen=True, ex=0.05))
    cmd = c.update(TagObs(seen=True, ex=0.05), RearRange(), 0.0)
    assert cmd.phase is Phase.REVERSE


def test_align_relost_tag_returns_to_search():
    c, _ = _ctrl(tag_lost_grace=0.5)
    c.start(0.0, TagObs(seen=True, ex=0.3))
    cmd = c.update(TagObs(seen=False), RearRange(), 2.0)   # lost past grace
    assert cmd.phase is Phase.SEARCH


# ── reverse ───────────────────────────────────────────────────────────────────
def _to_reverse(c):
    c.start(0.0, TagObs(seen=True, ex=0.0))
    c.update(TagObs(seen=True, ex=0.0), RearRange(0.5, 0.5), 0.0)   # ALIGN(centered) → REVERSE


def test_reverse_backs_straight_when_aligned():
    c, _ = _ctrl()
    _to_reverse(c)
    cmd = c.update(TagObs(seen=True, ex=0.0), RearRange(0.4, 0.4), 0.1)
    assert cmd.phase is Phase.REVERSE and cmd.linear_x < 0.0 and cmd.angular_z == 0.0


def test_reverse_stops_on_tof():
    c, _ = _ctrl()
    _to_reverse(c)
    cmd = c.update(TagObs(seen=True, ex=0.0), RearRange(0.09, 0.09), 0.2)
    assert cmd.done and cmd.linear_x == 0.0


def test_reverse_pulse_corrects_on_drift():
    c, _ = _ctrl(reverse_correct_tol=0.30)
    _to_reverse(c)
    cmd = c.update(TagObs(seen=True, ex=0.5), RearRange(0.4, 0.4), 0.2)   # drifted > tol
    assert cmd.phase is Phase.REVERSE and cmd.linear_x == 0.0 and cmd.angular_z != 0.0


def test_reverse_squares_up_near_wall():
    # near the wall with a rear ToF L/R imbalance → pulse-rotate to square (no translation)
    c, _ = _ctrl(square_engage_range=0.25, square_tol=0.02, stop_range=0.13)
    _to_reverse(c)
    cmd = c.update(TagObs(seen=True, ex=0.0), RearRange(0.24, 0.18), 0.2)  # delta 0.06 > tol
    assert cmd.phase is Phase.REVERSE and cmd.linear_x == 0.0 and cmd.angular_z != 0.0
    assert "square" in cmd.message


def test_reverse_square_sign_flips_direction():
    c1, _ = _ctrl(square_sign=1.0)
    _to_reverse(c1)
    a = c1.update(TagObs(seen=True, ex=0.0), RearRange(0.24, 0.18), 0.2).angular_z
    c2, _ = _ctrl(square_sign=-1.0)
    _to_reverse(c2)
    b = c2.update(TagObs(seen=True, ex=0.0), RearRange(0.24, 0.18), 0.2).angular_z
    assert a == -b and a != 0.0


def test_reverse_backs_straight_when_squared():
    # near the wall but already square (small delta) → straight in
    c, _ = _ctrl(square_engage_range=0.25, square_tol=0.02, stop_range=0.13)
    _to_reverse(c)
    cmd = c.update(TagObs(seen=True, ex=0.0), RearRange(0.20, 0.19), 0.2)  # delta 0.01 < tol
    assert cmd.phase is Phase.REVERSE and cmd.linear_x < 0.0 and cmd.angular_z == 0.0


def test_reverse_near_wall_backs_even_if_tag_lost():
    # this close in, heading comes from the ToF, not the tag — a tag dropout is fine
    c, _ = _ctrl(square_engage_range=0.25, stop_range=0.13)
    _to_reverse(c)
    cmd = c.update(TagObs(seen=False), RearRange(0.20, 0.19), 2.0)
    assert cmd.phase is Phase.REVERSE and cmd.linear_x < 0.0


def test_reverse_fails_when_tag_lost_and_nothing_near():
    c, _ = _ctrl()
    _to_reverse(c)
    cmd = c.update(TagObs(seen=False), RearRange(0.6, 0.6), 2.0)
    assert cmd.failed


def test_invalid_single_tof_still_stops_on_the_valid_one():
    c, _ = _ctrl()
    _to_reverse(c)
    cmd = c.update(TagObs(seen=True, ex=0.0), RearRange(0.08, None), 0.2)
    assert cmd.done


def test_stop_range_above_safety_proximity():
    assert DockConfig().stop_range > 0.075


def test_pulse_speed_above_rotation_floor():
    # a pulse must exceed the ~1.1 rad/s in-place rotation floor or the motors won't move
    assert DockConfig().pulse_speed > 1.1


# ── orient ──────────────────────────────────────────────────────────────────
def test_start_with_orient_target_begins_in_orient():
    c, _ = _ctrl()
    c.start(0.0, TagObs(seen=False), orient_target=1.0, heading=0.0)
    assert c.phase is Phase.ORIENT


def test_start_without_orient_target_is_unchanged():
    # backward compatible: no orient_target -> old ALIGN/SEARCH behavior
    c, _ = _ctrl()
    c.start(0.0, TagObs(seen=False))
    assert c.phase is Phase.SEARCH
    c2, _ = _ctrl()
    c2.start(0.0, TagObs(seen=True, ex=0.2))
    assert c2.phase is Phase.ALIGN


def test_orient_pulses_toward_target_both_ways():
    # target CCW of current heading (+err) -> +yaw; the reverse -> -yaw. No steer_sign.
    c, _ = _ctrl()
    c.start(0.0, TagObs(seen=False), orient_target=1.0, heading=0.0)
    cmd = c.update(TagObs(seen=False), RearRange(), 0.0, heading=0.0)
    assert cmd.phase is Phase.ORIENT and cmd.angular_z > 0.0
    c2, _ = _ctrl()
    c2.start(0.0, TagObs(seen=False), orient_target=-1.0, heading=0.0)
    cmd2 = c2.update(TagObs(seen=False), RearRange(), 0.0, heading=0.0)
    assert cmd2.angular_z < 0.0


def test_orient_hands_to_align_when_tag_appears():
    c, _ = _ctrl()
    c.start(0.0, TagObs(seen=False), orient_target=1.0, heading=0.0)
    cmd = c.update(TagObs(seen=True, ex=0.2), RearRange(), 0.1, heading=0.3)
    assert cmd.phase is Phase.ALIGN   # stops turning the moment the tag is in view


def test_orient_reaches_target_without_tag_falls_to_search():
    c, _ = _ctrl(orient_tol=0.1)
    c.start(0.0, TagObs(seen=False), orient_target=1.0, heading=0.0)
    cmd = c.update(TagObs(seen=False), RearRange(), 0.1, heading=1.0)  # err ~0
    assert cmd.phase is Phase.SEARCH


def test_orient_holds_when_heading_temporarily_unavailable():
    # A TF gap must NOT insta-fail to blind SEARCH — hold still, wait for TF
    # to come back. Permanent gap is caught by t_orient_max.
    c, _ = _ctrl()
    c.start(0.0, TagObs(seen=False), orient_target=1.0, heading=0.0)
    cmd = c.update(TagObs(seen=False), RearRange(), 0.1, heading=None)
    assert cmd.phase is Phase.ORIENT
    assert cmd.angular_z == 0.0
    # And once TF is back, it resumes turning normally.
    cmd = c.update(TagObs(seen=False), RearRange(), 0.5, heading=0.0)
    assert cmd.phase is Phase.ORIENT and cmd.angular_z > 0.0


def test_orient_entered_even_without_initial_heading():
    # start() enters ORIENT purely on orient_target — the phase tolerates the gap.
    c, _ = _ctrl()
    c.start(0.0, TagObs(seen=False), orient_target=1.0, heading=None)
    assert c.phase is Phase.ORIENT


def test_orient_timeout_falls_to_search():
    c, _ = _ctrl(t_orient_max=5.0)
    c.start(0.0, TagObs(seen=False), orient_target=3.0, heading=0.0)
    cmd = c.update(TagObs(seen=False), RearRange(), 6.0, heading=0.5)
    assert cmd.phase is Phase.SEARCH
