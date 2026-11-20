import numpy as np
import pytest

from app.pipeline.crossing import Detection
from app.pipeline.privacy import anonymise_frame, regions_to_blur


def textured(h=200, w=300, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def test_plate_band_and_driver_blurred_background_untouched():
    img = textured()
    det = Detection(1, "car", 0.9, (50, 50, 150, 150))
    out = anonymise_frame(img, [det])
    plate = (slice(120, 150), slice(50, 150))
    assert np.abs(out[plate].astype(int) - img[plate].astype(int)).mean() > 20
    assert out[plate].std() < img[plate].std()
    far = (slice(0, 40), slice(200, 300))
    assert np.array_equal(out[far], img[far])


def test_motorcycle_rider_head_region():
    regs = regions_to_blur(textured(), [Detection(1, "motorcycle", 0.9, (10, 10, 40, 110))])
    assert any(r[1] == 10 and r[3] >= 40 for r in regs)


def test_strong_mode_pixelates_whole_vehicle():
    img = textured()
    out = anonymise_frame(img, [Detection(1, "bus", 0.9, (20, 20, 120, 120))], "strong")
    assert out[20:120, 20:120].std() < img[20:120, 20:120].std() * 0.6


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        anonymise_frame(textured(), [], "none")


def test_plate_band_covers_mid_height_rear_plate():
    # Real rear view (eu1.jpg): the plate sat at 47% of the car box height.
    regs = regions_to_blur(textured(400, 400), [Detection(1, "car", 0.9, (0, 0, 400, 400))])
    covered = np.zeros(400, bool)
    for x, y, w, h, _ in regs:
        if x <= 0 and x + w >= 400:
            covered[max(0, y):y + h] = True
    assert covered[int(0.10 * 400):].all()


def test_face_detector_uses_dnn_when_model_present():
    from app.pipeline.privacy import FACE_MODEL, MODELS_DIR, FaceDetector
    fd = FaceDetector()
    if (MODELS_DIR / FACE_MODEL).exists():
        assert fd.name == "res10-ssd"
    assert fd.detect(np.zeros((200, 200, 3), np.uint8)) == []
    assert fd.detect(np.zeros((4, 4, 3), np.uint8)) == []


def test_blur_destroys_fine_detail_in_thin_band():
    img = np.zeros((40, 400, 3), np.uint8)
    img[:, ::4] = 255  # fine vertical stripes, like plate characters
    from app.pipeline.privacy import _blur
    _blur(img, 0, 0, 400, 40)
    assert img.std() < 10
