"""face_recognizer.py — SFace face recognition for the Jetson head_detector.

Resolves a detected face to a known name (from a gallery) or a stable per-session
'unknown_N', for publishing on /camera/identity. The Pi's behavior_node consumes
that id to key OMNI's per-person memory (SPEC Step 6: recognizer resolves, Pi
consumes).

FAIL-SAFE: construction and identify() swallow all errors and degrade to ''
(unknown). Face recognition must NEVER break person/face detection.

Gallery layout (offline enrollment):
    <gallery_dir>/<name>/*.jpg      (one folder per known person, 1+ photos)
    <gallery_dir>/<name>.jpg        (or a single photo named after the person)
Names are lowercased. Add photos and restart the node to (re)enroll.
"""
import collections
import glob
import os

import cv2
import numpy as np

# SFace's canonical cosine match threshold (OpenCV Zoo): >= => same person.
_DEFAULT_THRESHOLD = 0.363

# Quality gate defaults. A turned/tiny/low-confidence face embeds poorly, so it
# won't match its own gallery entry — without this gate every such frame minted a
# brand-new junk unknown_N.
# Calibrated against real faces: a good frontal face measures ~0.0-0.35 nose offset
# (a mildly-turned but perfectly usable face reads ~0.34), while a real profile —
# nose level with an eye — is ~0.5+. 0.40 accepts usable frontal faces and rejects
# turned ones. Note the gate only blocks REGISTERING a new identity; matching an
# already-known face still works at any angle.
_MIN_SCORE = 0.80      # YuNet detection confidence (node only reports >=0.6 at all)
_MIN_FACE_PX = 80      # min face box width/height in capture pixels
_MAX_NOSE_OFF = 0.40   # |nose - eye centre| / eye separation; larger => turned head


def _cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return -1.0
    return float(np.dot(a, b) / (na * nb))


def face_quality_ok(raw_face, *, min_score=_MIN_SCORE, min_px=_MIN_FACE_PX,
                    max_nose_off=_MAX_NOSE_OFF):
    """True when a YuNet row is a big, confident, roughly FRONTAL face — i.e. good
    enough to enrol or to register as a new identity.

    YuNet row: [x, y, w, h, rex,rey, lex,ley, nx,ny, rmx,rmy, lmx,lmy, score].
    Frontality proxy: on a frontal face the nose sits near the midpoint of the eyes;
    as the head turns the nose slides toward one eye.
    """
    try:
        w, h, score = float(raw_face[2]), float(raw_face[3]), float(raw_face[14])
        if score < min_score or w < min_px or h < min_px:
            return False
        rex, lex, nx = float(raw_face[4]), float(raw_face[6]), float(raw_face[8])
        eye_sep = abs(lex - rex)
        if eye_sep < 1.0:
            return False
        return abs(nx - (lex + rex) / 2.0) / eye_sep <= max_nose_off
    except (IndexError, TypeError, ValueError):
        return False


class IdentitySmoother:
    """Majority-vote hysteresis over recent frames.

    Raw per-frame recognition is noisy — one off-angle or blank frame would flip the
    published id (rafael -> '' -> unknown_24 -> rafael). We only switch the published
    identity once a candidate dominates a sliding window, so brief blips are ignored
    while a real change (person leaves / swaps) still lands within ~a second.
    """

    def __init__(self, window=15, switch_ratio=0.6):
        self._buf = collections.deque(maxlen=max(1, int(window)))
        self._ratio = float(switch_ratio)
        self._current = ''

    @property
    def current(self):
        return self._current

    def update(self, raw_id):
        self._buf.append(raw_id or '')
        # Need a reasonable sample before trusting a switch.
        if len(self._buf) < max(2, self._buf.maxlen // 2):
            return self._current
        top, n = collections.Counter(self._buf).most_common(1)[0]
        if top != self._current and n >= self._ratio * len(self._buf):
            self._current = top
        return self._current


def _box_iou(a, b):
    """Intersection-over-union of two face dicts ({x1,y1,x2,y2,bw,bh})."""
    ix1, iy1 = max(a['x1'], b['x1']), max(a['y1'], b['y1'])
    ix2, iy2 = min(a['x2'], b['x2']), min(a['y2'], b['y2'])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a['bw'] * a['bh'] + b['bw'] * b['bh'] - inter
    return inter / union if union > 0 else 0.0


def clean_person_name(value):
    """Normalise an enrol name, or return '' if it isn't usable as one.

    A name becomes a DIRECTORY under the gallery dir, so this is a safety boundary as
    much as a tidy-up: anything with a path separator, '..', or other punctuation is
    rejected rather than sanitised, so a malformed request can never write outside the
    gallery or enrol a junk label like '[1,2,3]'.
    """
    text = str(value or '').strip().lower()
    if not text or len(text) > 64:
        return ''
    if not all(c.isalnum() or c in (' ', '_', '-') for c in text):
        return ''
    if not any(c.isalnum() for c in text):
        return ''
    # 'unknown' is the anonymous id space (unknown_N), not a person. The model will
    # happily call remember_person("unknown") as a placeholder when it never caught a
    # name — that created a real gallery entry named 'unknown' holding a real face.
    if text == 'unknown' or text.startswith('unknown_'):
        return ''
    return text


def parse_enroll_request(data):
    """Parse a /camera/enroll_request payload into (name, target).

    Accepts either a bare name — legacy form, meaning 'enrol the primary face' —
    or JSON naming WHICH visible face to learn:
        "alice"                              -> ('alice', None)
        {"name": "alice", "target": "unknown_3"}  -> ('alice', 'unknown_3')
        {"name": "alice", "track": 2}            -> ('alice', 2)
    A malformed payload yields ('', None), which callers reject as an empty name.
    """
    import json as _json
    text = (data or '').strip()
    if not text.startswith('{'):
        return clean_person_name(text), None
    try:
        obj = _json.loads(text)
    except ValueError:
        return '', None
    if not isinstance(obj, dict):
        return '', None
    name = clean_person_name(obj.get('name'))
    if obj.get('track') is not None:
        try:
            return name, int(obj['track'])
        except (TypeError, ValueError):
            return name, None
    target = obj.get('target')
    return name, (str(target).strip().lower() if target else None)


def resolve_target(rows, target):
    """Pick the face row a caller means: an identity (str), a track id (int), or —
    when target is None — the primary (largest) face.

    Returns None when the requested person is NOT among `rows`. Callers must treat
    None as 'refuse'; substituting the closest face is precisely the wrong-person
    enrolment bug this lookup exists to prevent.

    Lives at module level because both the tracker and the node's enrol path (which
    works on a locked snapshot of rows) need exactly this rule.
    """
    if not rows:
        return None
    if target is None or target == '':
        return rows[0]                      # rows are largest-face-first
    key = 'track' if isinstance(target, int) and not isinstance(target, bool) else 'identity'
    for row in rows:
        if row[key] == target:
            return row
    return None


class _Track:
    """One face followed across frames, with its OWN identity vote."""

    __slots__ = ('id', 'face', 'smoother', 'missed', 'raw')

    def __init__(self, tid, face, window, switch_ratio):
        self.id = tid
        self.face = face
        self.smoother = IdentitySmoother(window=window, switch_ratio=switch_ratio)
        self.missed = 0
        self.raw = ''


class FaceTracker:
    """Associates faces frame-to-frame by IoU and smooths each track separately.

    Why per-track: a single global IdentitySmoother implicitly assumes ONE face in
    frame. With two people the votes interleave in one window, so whoever happens to
    be recognised more often that second wins — a second person walking past could
    flip the published identity of the person actually talking. Each track now votes
    only on its own recognitions.

    This also gives every visible face a stable handle, which is what lets enrolment
    name a SPECIFIC person instead of whoever is closest to the camera.

    The identify step is injected (identify_fn) so this class stays free of SFace and
    can be unit-tested without models or a camera.
    """

    def __init__(self, iou_threshold=0.3, max_missed=8, window=15, switch_ratio=0.6,
                 primary_hold=12):
        self._iou_thr = float(iou_threshold)
        self._max_missed = int(max_missed)
        self._window = int(window)
        self._ratio = float(switch_ratio)
        self._primary_hold = int(primary_hold)
        self._tracks = []          # list[_Track], live
        self._next_id = 1
        self._latest = []          # last update()'s result rows, largest face first
        # Last confident primary id + how many frames we've coasted on it. A face
        # whose box jumps far enough to fail IoU lands on a NEW track with an empty
        # vote; without this the published identity would blink to '' on every such
        # handover (seen live as short-lived tracks beside a stable one).
        self._held_id = ''
        self._held_frames = 0

    def update(self, faces, identify_fn):
        """Advance one frame.

        faces: list of face dicts (largest-first order is imposed here, not assumed).
        identify_fn(face) -> raw id string for that face this frame.

        Returns rows [{'track', 'identity', 'raw', 'face', 'primary'}], largest face
        first; 'identity' is that track's SMOOTHED id, 'raw' this frame's unsmoothed
        one. The single largest face is flagged primary — that is the one whose id
        goes out on /camera/identity, preserving the pre-multi-face contract.
        """
        ordered = sorted(faces, key=lambda f: -(f['bw'] * f['bh']))
        unmatched = list(self._tracks)
        rows = []

        for face in ordered:
            best, best_iou = None, self._iou_thr
            for track in unmatched:
                iou = _box_iou(face, track.face)
                if iou >= best_iou:
                    best, best_iou = track, iou
            if best is None:
                best = _Track(self._next_id, face, self._window, self._ratio)
                self._next_id += 1
                self._tracks.append(best)
            else:
                unmatched.remove(best)
            best.face = face
            best.missed = 0
            try:
                best.raw = identify_fn(face) or ''
            except Exception:  # noqa: BLE001 - a bad frame must not kill tracking
                best.raw = ''
            rows.append({'track': best.id, 'identity': best.smoother.update(best.raw),
                         'raw': best.raw, 'face': face, 'primary': False})

        # Tracks nobody matched this frame: tolerate a short dropout (a blink, a turn)
        # before forgetting them, so a track id survives brief detection gaps.
        for track in unmatched:
            track.missed += 1
        self._tracks = [t for t in self._tracks if t.missed <= self._max_missed]

        if rows:
            rows[0]['primary'] = True
        self._latest = rows

        # Carry the last confident primary id across a track handover, but only for a
        # bounded number of frames so a person who actually leaves is released.
        primary = rows[0]['identity'] if rows else ''
        if primary:
            self._held_id, self._held_frames = primary, 0
        elif self._held_id:
            self._held_frames += 1
            if self._held_frames > self._primary_hold:
                self._held_id, self._held_frames = '', 0
        return rows

    def primary_identity(self):
        """Smoothed id of the largest face, or '' when nobody is resolved.

        Briefly holds the previous id across a track handover (see _held_id) so this —
        the value published on /camera/identity — stays as steady as it was before
        faces were tracked individually.
        """
        if self._latest and self._latest[0]['identity']:
            return self._latest[0]['identity']
        return self._held_id

    def find(self, target):
        """Locate a currently-visible face by identity (str) or track id (int).

        Returns its row, or None when that person is not in view — callers must treat
        None as 'refuse', never as 'fall back to the closest face'. Unlike
        resolve_target(), an empty target matches nothing here: find() is always a
        lookup for someone SPECIFIC.
        """
        if target is None or target == '':
            return None
        return resolve_target(self._latest, target)

    @property
    def rows(self):
        """Last frame's rows (largest face first)."""
        return list(self._latest)


class FaceRecognizer:
    def __init__(self, sface_model, yunet_model, gallery_dir, logger,
                 *, unknown_dir=None, match_threshold=_DEFAULT_THRESHOLD,
                 unknown_threshold=_DEFAULT_THRESHOLD, min_score=_MIN_SCORE,
                 min_face_px=_MIN_FACE_PX, max_nose_off=_MAX_NOSE_OFF):
        self._log = logger
        self._ok = False
        self._match_thr = float(match_threshold)
        self._unknown_thr = float(unknown_threshold)
        self._min_score = float(min_score)
        self._min_face_px = float(min_face_px)
        self._max_nose_off = float(max_nose_off)
        self._gallery = {}       # name -> list[np.ndarray] reference embeddings
        # Persisted unknowns: id(int) -> list[np.ndarray]. Stable across reboots
        # (crops saved under <unknown_dir>/unknown_<id>/ and re-embedded at boot).
        self._unknowns = {}
        self._next_unknown = 1
        self._gallery_dir = gallery_dir
        self._unknown_dir = unknown_dir or os.path.join(
            os.path.dirname(gallery_dir.rstrip("/")) or ".", "unknown_faces")
        try:
            self._sf = cv2.FaceRecognizerSF.create(sface_model, "")
            # Own YuNet just for enrolling gallery/unknown images (the live path
            # passes in the capture-thread YuNet's raw rows directly).
            self._yn = cv2.FaceDetectorYN.create(yunet_model, "", (320, 320), 0.6, 0.3, 5000)
            self._load_gallery(gallery_dir)
            self._load_unknowns()
            self.consolidate()
            self._ok = True
            self._log.info(
                f"FaceRecognizer ready — {len(self._gallery)} known "
                f"person(s): {sorted(self._gallery)}; "
                f"{len(self._unknowns)} persisted unknown(s)"
            )
        except Exception as e:  # noqa: BLE001
            self._log.warn(f"FaceRecognizer init failed ({e}) — identity disabled")

    @property
    def ok(self):
        return self._ok

    # ── enrollment ──────────────────────────────────────────────────────────────

    def _embed_file(self, path):
        """Detect + align + embed a stored image. Returns None if no face is found."""
        img = cv2.imread(path)
        if img is None:
            return None
        h, w = img.shape[:2]
        self._yn.setInputSize((w, h))
        _, faces = self._yn.detect(img)
        if faces is None or len(faces) == 0:
            return None
        faces = sorted(faces, key=lambda f: -(f[2] * f[3]))  # largest face
        return self._sf.feature(self._sf.alignCrop(img, faces[0])).flatten()

    @staticmethod
    def _feat_path(img_path):
        return os.path.splitext(img_path)[0] + ".npy"

    def _embed_ref(self, path):
        """Embedding for a stored reference, preferring a cached .npy.

        We persist the ALIGNED 112x112 crop, which is a tight face fill with no
        margin — YuNet cannot reliably re-detect a face in it, so re-embedding at
        boot silently loses references (observed: 7 of 13 stored unknowns failed to
        reload). For a named person that would mean quietly no longer recognising
        them, with no error. Caching the embedding removes the re-detection step
        entirely; the crop is kept only for human inspection. Older crops without a
        cached vector are re-embedded once and the cache written, so this self-heals.
        """
        npy = self._feat_path(path)
        if os.path.isfile(npy):
            try:
                return np.load(npy)
            except Exception as e:  # noqa: BLE001 - fall back to re-embedding
                self._log.warn(f"FaceRecognizer: unreadable cache {npy} ({e})")
        feat = self._embed_file(path)
        if feat is not None:
            try:
                np.save(npy, feat)
            except Exception:  # noqa: BLE001 - caching is best-effort
                pass
        return feat

    def _load_gallery(self, gallery_dir):
        if not gallery_dir or not os.path.isdir(gallery_dir):
            return
        for entry in sorted(os.listdir(gallery_dir)):
            path = os.path.join(gallery_dir, entry)
            name, files = None, []
            if os.path.isdir(path):
                name = entry.lower()
                for ext in ("jpg", "jpeg", "png"):
                    files += glob.glob(os.path.join(path, f"*.{ext}"))
            elif entry.lower().rsplit(".", 1)[-1] in ("jpg", "jpeg", "png"):
                name = entry.rsplit(".", 1)[0].lower()
                files = [path]
            if not name:
                continue
            embs = [e for e in (self._embed_ref(f) for f in sorted(files)) if e is not None]
            if embs:
                self._gallery[name] = embs
            else:
                self._log.warn(f"FaceRecognizer: no usable face in gallery entry '{entry}'")

    def _load_unknowns(self):
        """Re-load persisted unknowns from <unknown_dir>/unknown_<id>/*.jpg so
        unknown ids stay stable across reboots."""
        if not os.path.isdir(self._unknown_dir):
            return
        for entry in sorted(os.listdir(self._unknown_dir)):
            path = os.path.join(self._unknown_dir, entry)
            if not (os.path.isdir(path) and entry.startswith("unknown_")):
                continue
            try:
                uid = int(entry.split("_", 1)[1])
            except (ValueError, IndexError):
                continue
            files = []
            for ext in ("jpg", "jpeg", "png"):
                files += glob.glob(os.path.join(path, f"*.{ext}"))
            embs = [e for e in (self._embed_ref(f) for f in sorted(files)) if e is not None]
            if embs:
                self._unknowns[uid] = embs
                self._next_unknown = max(self._next_unknown, uid + 1)

    def _save_crop(self, base_dir, label, aligned, feat=None):
        """Persist an aligned face crop under base_dir/label/<ts>.jpg, plus its
        embedding as a sibling .npy so reloading never depends on re-detecting a
        face in the tight crop (see _embed_ref). Returns the image path or None."""
        try:
            import time as _t
            d = os.path.join(base_dir, label)
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, f"{int(_t.time() * 1000)}.jpg")
            cv2.imwrite(path, aligned)
            if feat is not None:
                try:
                    np.save(self._feat_path(path), feat)
                except Exception:  # noqa: BLE001 - cache is an optimisation
                    pass
            return path
        except Exception as e:  # noqa: BLE001
            self._log.warn(f"FaceRecognizer: crop save failed for {label}: {e}")
            return None

    # ── live recognition ────────────────────────────────────────────────────────

    def identify(self, frame, raw_face, register=True):
        """Resolve a face (a raw YuNet row, [15]) in `frame` to a person id.
        Returns a known name, a stable 'unknown_N', or '' on any failure.

        register=False resolves against people we already know but never mints a new
        unknown_N. Callers pass False for non-primary faces: once every visible face
        is identified (not just the closest), marginal and spurious detections would
        otherwise each mint a junk identity — measured at ~10 new unknowns in 4
        minutes with a single real person in the room.
        """
        if not self._ok:
            return ""
        try:
            feat = self._sf.feature(self._sf.alignCrop(frame, raw_face)).flatten()
        except Exception as e:  # noqa: BLE001
            self._log.warn(f"FaceRecognizer.identify error: {e}", throttle_duration_sec=5.0)
            return ""

        # Best match among known people.
        best_name, best_sim = None, -1.0
        for name, refs in self._gallery.items():
            sim = max(_cosine(feat, r) for r in refs)
            if sim > best_sim:
                best_name, best_sim = name, sim
        if best_name is not None and best_sim >= self._match_thr:
            return best_name

        # Otherwise match a persisted unknown (stable across reboots).
        for uid, refs in self._unknowns.items():
            if max(_cosine(feat, r) for r in refs) >= self._unknown_thr:
                return f"unknown_{uid}"

        # No match. Only mint a NEW unknown for a good, frontal face — a turned or
        # distant face embeds poorly and would otherwise spawn endless junk ids —
        # and only for the primary face (see `register`).
        if not register or not self.quality_ok(raw_face):
            return ""
        uid = self._next_unknown
        self._next_unknown += 1
        self._unknowns[uid] = [feat]
        try:
            self._save_crop(self._unknown_dir, f"unknown_{uid}",
                            self._sf.alignCrop(frame, raw_face), feat)
        except Exception as e:  # noqa: BLE001 - in-memory unknown still registered
            self._log.warn(f"FaceRecognizer: unknown_{uid} crop save failed: {e}")
        return f"unknown_{uid}"

    def consolidate(self):
        """Tidy the persisted unknowns against the current gallery. Run at startup.

        Two kinds of stale entry accumulate:
          * Unknowns that ARE a now-enrolled person. A person registers as unknown_N
            before enrolment (or at a pose their early references missed); as their
            gallery grows those entries start matching them, but merging only ran
            during an enrolment, so they lingered and kept being published INSTEAD of
            the person's name. Observed: three unknowns scoring 0.40-0.44 against a
            gallery whose match bar is 0.363.
          * Folders whose crops yield no embedding at all (pre-embedding-cache junk),
            which can never match anything.

        Returns (merged, pruned) counts. Only ever removes anonymous entries — named
        gallery people are never touched.
        """
        import shutil
        merged = pruned = 0
        if not os.path.isdir(self._unknown_dir):
            return (0, 0)

        # 1) Unknowns that now match a known person are redundant — drop them.
        for uid in sorted(self._unknowns):
            refs = self._unknowns[uid]
            best_name, best = None, -1.0
            for name, grefs in self._gallery.items():
                sim = max(_cosine(r, g) for r in refs for g in grefs)
                if sim > best:
                    best_name, best = name, sim
            if best_name is not None and best >= self._match_thr:
                self._unknowns.pop(uid, None)
                shutil.rmtree(os.path.join(self._unknown_dir, f"unknown_{uid}"),
                              ignore_errors=True)
                merged += 1
                self._log.info(
                    f"FaceRecognizer: consolidated unknown_{uid} into "
                    f"'{best_name}' (cos {best:.3f})"
                )

        # 2) Unusable folders — present on disk but produced no embedding on load.
        for entry in sorted(os.listdir(self._unknown_dir)):
            path = os.path.join(self._unknown_dir, entry)
            if not (os.path.isdir(path) and entry.startswith("unknown_")):
                continue
            try:
                uid = int(entry.split("_", 1)[1])
            except (ValueError, IndexError):
                continue
            if uid not in self._unknowns:
                shutil.rmtree(path, ignore_errors=True)
                pruned += 1
        if pruned:
            self._log.info(f"FaceRecognizer: pruned {pruned} unusable unknown folder(s)")
        return (merged, pruned)

    def quality_ok(self, raw_face):
        """Is this face good enough to enrol / register as a new identity?"""
        return face_quality_ok(raw_face, min_score=self._min_score,
                               min_px=self._min_face_px,
                               max_nose_off=self._max_nose_off)

    # ── on-the-fly enrollment (Step 6) ──────────────────────────────────────────

    def enroll(self, name, frame, raw_face):
        """Learn the given face as `name`: add its embedding to the live gallery
        and persist an aligned crop under <gallery_dir>/<name>/ so it survives a
        restart (re-embedded at boot). Returns True on success."""
        if not self._ok or not name:
            return False
        try:
            aligned = self._sf.alignCrop(frame, raw_face)
            feat = self._sf.feature(aligned).flatten()
        except Exception as e:  # noqa: BLE001
            self._log.warn(f"FaceRecognizer.enroll embed error: {e}")
            return False
        self._gallery.setdefault(name, []).append(feat)  # recognised immediately
        path = self._save_crop(self._gallery_dir, name, aligned, feat)
        self._log.info(f"FaceRecognizer: enrolled '{name}'" + (f" -> {path}" if path else " (memory only)"))
        # This face is no longer anonymous — drop any matching persisted unknown so
        # it isn't double-counted. (behavior_node re-keys that unknown's memories.)
        import shutil
        for uid in [u for u, refs in self._unknowns.items()
                    if max(_cosine(feat, r) for r in refs) >= self._match_thr]:
            self._unknowns.pop(uid, None)
            shutil.rmtree(os.path.join(self._unknown_dir, f"unknown_{uid}"), ignore_errors=True)
            self._log.info(f"FaceRecognizer: merged unknown_{uid} into '{name}'")
        return True
