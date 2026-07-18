"""Unit tests for FaceTracker — multi-face association + per-track identity smoothing.

Runs on the Pi (no TensorRT, no models): FaceTracker is pure Python/numpy and the
identify step is injected, so these tests never touch SFace or a camera.

    cd ~/omni_ws/jetson/head_detector && python3 -m pytest tests/ -v
"""
import pytest

from head_detector.face_recognizer import (FaceTracker, parse_enroll_request,
                                           resolve_target)


def face(x, y, w, h):
    """A face dict in the shape YuNetFace.detect() produces (publish space)."""
    return {'x1': float(x), 'y1': float(y), 'x2': float(x + w), 'y2': float(y + h),
            'cx': x + w / 2.0, 'cy': y + h / 2.0, 'bw': float(w), 'bh': float(h),
            '_raw': None}


def const(mapping):
    """identify_fn that resolves a face to an id by its box width — a cheap way for a
    test to say 'this particular face is rafael' without any real recognition."""
    return lambda f: mapping.get(int(f['bw']), '')


class TestAssociation:
    def test_same_face_keeps_its_track_across_frames(self):
        t = FaceTracker()
        ids = []
        for dx in range(0, 20, 4):          # face drifts slowly across the frame
            out = t.update([face(100 + dx, 100, 90, 90)], const({90: 'rafael'}))
            ids.append(out[0]['track'])
        assert len(set(ids)) == 1, f'track id changed while drifting: {ids}'

    def test_two_faces_get_distinct_stable_tracks(self):
        t = FaceTracker()
        seen = []
        for _ in range(6):
            out = t.update([face(100, 100, 120, 120), face(500, 100, 80, 80)],
                           const({120: 'rafael', 80: 'unknown_3'}))
            seen.append(tuple(sorted(d['track'] for d in out)))
        assert len(set(seen)) == 1, f'track ids unstable across frames: {seen}'
        assert len(set(seen.pop())) == 2, 'two faces should hold two distinct tracks'

    def test_a_jump_beyond_iou_threshold_starts_a_new_track(self):
        t = FaceTracker()
        first = t.update([face(100, 100, 90, 90)], const({90: 'a'}))[0]['track']
        second = t.update([face(900, 500, 90, 90)], const({90: 'a'}))[0]['track']
        assert first != second

    def test_track_expires_after_max_missed_frames(self):
        t = FaceTracker(max_missed=3)
        first = t.update([face(100, 100, 90, 90)], const({90: 'a'}))[0]['track']
        for _ in range(4):                  # face gone for longer than max_missed
            t.update([], const({}))
        reappeared = t.update([face(100, 100, 90, 90)], const({90: 'a'}))[0]['track']
        assert reappeared != first, 'an expired track should not be reused'

    def test_brief_dropout_within_max_missed_keeps_the_track(self):
        t = FaceTracker(max_missed=5)
        first = t.update([face(100, 100, 90, 90)], const({90: 'a'}))[0]['track']
        t.update([], const({}))             # one blank frame — a blink, not a departure
        again = t.update([face(100, 100, 90, 90)], const({90: 'a'}))[0]['track']
        assert again == first


class TestPerTrackSmoothing:
    """The reason this class exists: one global smoother mixed votes from different
    faces, so a second person in frame could flip the first person's published id."""

    def test_two_faces_do_not_contaminate_each_others_identity(self):
        t = FaceTracker(window=8, switch_ratio=0.6)
        big, small = face(100, 100, 200, 200), face(700, 100, 90, 90)
        out = None
        for _ in range(12):
            out = t.update([big, small], const({200: 'rafael', 90: 'alice'}))
        by_id = {d['identity']: d for d in out}
        assert set(by_id) == {'rafael', 'alice'}, f'identities bled together: {out}'

    def test_a_single_bad_frame_does_not_flip_a_track(self):
        t = FaceTracker(window=10, switch_ratio=0.6)
        f = face(100, 100, 90, 90)
        for _ in range(10):
            t.update([f], const({90: 'rafael'}))
        out = t.update([f], lambda _f: '')      # one blank recognition frame
        assert out[0]['identity'] == 'rafael', 'hysteresis should absorb one bad frame'

    def test_a_sustained_change_does_land(self):
        t = FaceTracker(window=8, switch_ratio=0.6)
        f = face(100, 100, 90, 90)
        for _ in range(10):
            t.update([f], const({90: 'rafael'}))
        out = None
        for _ in range(10):                     # person actually swapped seats
            out = t.update([f], const({90: 'alice'}))
        assert out[0]['identity'] == 'alice'


class TestPrimaryAndOrdering:
    def test_largest_face_is_primary_and_sorts_first(self):
        t = FaceTracker(window=6, switch_ratio=0.6)
        for _ in range(8):      # smoothing withholds a verdict until it has samples
            out = t.update([face(700, 100, 90, 90), face(100, 100, 250, 250)],
                           const({90: 'alice', 250: 'rafael'}))
        assert out[0]['primary'] is True
        assert out[0]['identity'] == 'rafael', 'primary must be the LARGEST face'
        assert all(d['primary'] is False for d in out[1:]), 'only one primary'

    def test_primary_matches_the_legacy_single_face_behaviour(self):
        """/camera/identity still carries exactly what it carried before."""
        t = FaceTracker()
        for _ in range(10):
            out = t.update([face(100, 100, 120, 120)], const({120: 'rafael'}))
        assert t.primary_identity() == 'rafael' == out[0]['identity']

    def test_no_faces_yields_empty_primary(self):
        t = FaceTracker()
        assert t.update([], const({})) == []
        assert t.primary_identity() == ''


class TestTargetLookup:
    """Backing the enrolment fix: find a named face without assuming it is closest."""

    def test_find_by_identity_returns_that_face_not_the_primary(self):
        t = FaceTracker(window=6, switch_ratio=0.6)
        big, small = face(100, 100, 250, 250), face(700, 100, 90, 90)
        for _ in range(10):
            t.update([big, small], const({250: 'rafael', 90: 'unknown_3'}))
        hit = t.find('unknown_3')
        assert hit is not None
        assert hit['identity'] == 'unknown_3'
        assert hit['primary'] is False, 'the target here is deliberately NOT the closest'
        assert hit['face']['bw'] == 90

    def test_find_by_track_id(self):
        t = FaceTracker()
        out = t.update([face(100, 100, 120, 120)], const({120: 'rafael'}))
        assert t.find(out[0]['track'])['identity'] == out[0]['identity']

    def test_find_returns_none_when_the_target_is_absent(self):
        t = FaceTracker()
        t.update([face(100, 100, 120, 120)], const({120: 'rafael'}))
        assert t.find('nobody_here') is None
        assert t.find(999) is None


class TestParseEnrollRequest:
    """The wire format for 'learn THIS person', kept backward compatible."""

    def test_bare_name_is_the_legacy_primary_face_form(self):
        assert parse_enroll_request('Rafael') == ('rafael', None)

    def test_json_target_names_an_identity(self):
        assert parse_enroll_request('{"name": "Alice", "target": "unknown_3"}') \
            == ('alice', 'unknown_3')

    def test_json_track_names_a_track_id(self):
        assert parse_enroll_request('{"name": "alice", "track": 2}') == ('alice', 2)

    def test_track_wins_when_both_are_given(self):
        got = parse_enroll_request('{"name": "a", "track": 7, "target": "unknown_1"}')
        assert got == ('a', 7)

    @pytest.mark.parametrize('payload', [
        '{"name": "a", "track": "not-an-int"}',   # unusable track -> primary face
        '{"name": "a"}',                          # no target at all
        '{"name": "a", "target": null}',
    ])
    def test_missing_or_unusable_target_falls_back_to_primary(self, payload):
        assert parse_enroll_request(payload) == ('a', None)

    @pytest.mark.parametrize('payload', ['{bad json', '{"target": "unknown_1"}',
                                         '[1,2,3]', '', None, '   '])
    def test_malformed_payloads_yield_an_empty_name(self, payload):
        # An empty name is what the node rejects — a bad payload must never enrol.
        assert parse_enroll_request(payload)[0] == ''

    @pytest.mark.parametrize('bad', ['../../etc/passwd', 'a/b', 'a\\b', '..',
                                     'x' * 65, '!!!', '{"name": "../evil"}'])
    def test_names_that_could_escape_the_gallery_dir_are_rejected(self, bad):
        # The name becomes a directory under the gallery — never let it traverse.
        assert parse_enroll_request(bad)[0] == ''

    @pytest.mark.parametrize('good,want', [('Rafael', 'rafael'), ('Mary Jane', 'mary jane'),
                                           ('jean-luc', 'jean-luc'), ('user_2', 'user_2')])
    def test_ordinary_names_still_pass(self, good, want):
        assert parse_enroll_request(good)[0] == want


class TestResolveTarget:
    def rows(self):
        return [{'track': 1, 'identity': 'rafael', 'primary': True},
                {'track': 2, 'identity': 'unknown_3', 'primary': False}]

    def test_no_target_means_the_primary_face(self):
        assert resolve_target(self.rows(), None)['identity'] == 'rafael'

    def test_target_selects_a_non_primary_face(self):
        assert resolve_target(self.rows(), 'unknown_3')['track'] == 2

    def test_absent_target_returns_none_and_never_the_primary(self):
        """The wrong-person bug in one assertion: an off-screen target must NOT
        silently resolve to whoever happens to be closest to the camera."""
        assert resolve_target(self.rows(), 'alice') is None

    def test_empty_rows_resolve_to_none(self):
        assert resolve_target([], None) is None
        assert resolve_target([], 'rafael') is None


class TestPrimaryHold:
    """A face whose box jumps far enough to fail IoU starts a NEW track with an empty
    vote. Observed live: brief spurious tracks appeared beside a stable one. Without a
    hold, /camera/identity would blip to '' — the exact flicker smoothing exists to
    stop, and a regression against the old single-smoother behaviour."""

    def settled(self, t, name='rafael'):
        for _ in range(10):
            t.update([face(100, 100, 90, 90)], const({90: name}))
        assert t.primary_identity() == name

    def test_a_new_track_does_not_blank_the_published_identity(self):
        t = FaceTracker(window=8, switch_ratio=0.6, primary_hold=10)
        self.settled(t)
        # Box jumps -> fails IoU -> fresh track, no verdict yet.
        t.update([face(900, 500, 90, 90)], const({90: 'rafael'}))
        assert t.primary_identity() == 'rafael', 'a track handover must not blank the id'

    def test_the_hold_expires_once_the_person_is_really_gone(self):
        t = FaceTracker(window=8, switch_ratio=0.6, primary_hold=5)
        self.settled(t)
        for _ in range(8):
            t.update([], const({}))
        assert t.primary_identity() == '', 'the hold must not pin a departed person'

    def test_a_confident_new_identity_replaces_the_held_one(self):
        t = FaceTracker(window=6, switch_ratio=0.6, primary_hold=20)
        self.settled(t)
        out = None
        for _ in range(10):
            out = t.update([face(900, 500, 90, 90)], const({90: 'alice'}))
        assert out[0]['identity'] == 'alice'
        assert t.primary_identity() == 'alice', 'hold must yield to a real verdict'


class TestRegisterGate:
    """identify(register=False) must still MATCH, but never MINT.

    Guards the regression measured live: identifying every visible face instead of
    only the closest one minted ~10 junk unknowns in 4 minutes with one real person
    present, because marginal secondary detections each registered as new.
    """

    class FakeRecognizer:
        """Stands in for FaceRecognizer: the real one needs SFace + a gallery."""

        def __init__(self):
            self.minted = 0

        def identify(self, frame, raw_face, register=True):
            if raw_face == 'known':
                return 'rafael'
            if not register:
                return ''
            self.minted += 1
            return f'unknown_{self.minted}'

    def test_non_primary_unknown_face_does_not_mint(self):
        r = self.FakeRecognizer()
        assert r.identify(None, 'stranger', register=False) == ''
        assert r.minted == 0

    def test_primary_unknown_face_still_mints(self):
        r = self.FakeRecognizer()
        assert r.identify(None, 'stranger', register=True) == 'unknown_1'
        assert r.minted == 1

    def test_a_known_face_is_matched_even_when_not_registering(self):
        r = self.FakeRecognizer()
        assert r.identify(None, 'known', register=False) == 'rafael'
        assert r.minted == 0, 'matching must never mint'


class TestReservedNames:
    """'unknown' is the anonymous id space, not a person. Observed live: the model
    called remember_person("unknown") as a placeholder and a real gallery entry named
    'unknown' was created holding a real person's face."""

    @pytest.mark.parametrize('bad', ['unknown', 'Unknown', 'UNKNOWN',
                                     'unknown_3', 'unknown_12'])
    def test_unknown_is_rejected_as_a_person_name(self, bad):
        assert parse_enroll_request(bad)[0] == ''

    def test_json_form_rejects_it_too(self):
        assert parse_enroll_request('{"name": "unknown"}')[0] == ''

    def test_names_merely_containing_unknown_are_fine(self):
        assert parse_enroll_request('unknowned')[0] == 'unknowned'
