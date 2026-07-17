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
            embs = [e for e in (self._embed_file(f) for f in sorted(files)) if e is not None]
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
            embs = [e for e in (self._embed_file(f) for f in sorted(files)) if e is not None]
            if embs:
                self._unknowns[uid] = embs
                self._next_unknown = max(self._next_unknown, uid + 1)

    def _save_crop(self, base_dir, label, aligned):
        """Persist an aligned face crop under base_dir/label/<ts>.jpg; return path/None."""
        try:
            import time as _t
            d = os.path.join(base_dir, label)
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, f"{int(_t.time() * 1000)}.jpg")
            cv2.imwrite(path, aligned)
            return path
        except Exception as e:  # noqa: BLE001
            self._log.warn(f"FaceRecognizer: crop save failed for {label}: {e}")
            return None

    # ── live recognition ────────────────────────────────────────────────────────

    def identify(self, frame, raw_face):
        """Resolve a face (a raw YuNet row, [15]) in `frame` to a person id.
        Returns a known name, a stable 'unknown_N', or '' on any failure."""
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
        # distant face embeds poorly and would otherwise spawn endless junk ids.
        if not self.quality_ok(raw_face):
            return ""
        uid = self._next_unknown
        self._next_unknown += 1
        self._unknowns[uid] = [feat]
        try:
            self._save_crop(self._unknown_dir, f"unknown_{uid}",
                            self._sf.alignCrop(frame, raw_face))
        except Exception as e:  # noqa: BLE001 - in-memory unknown still registered
            self._log.warn(f"FaceRecognizer: unknown_{uid} crop save failed: {e}")
        return f"unknown_{uid}"

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
        path = self._save_crop(self._gallery_dir, name, aligned)
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
