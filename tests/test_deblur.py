import cv2
import numpy as np
import pandas as pd
import pytest

from alpr import deblur

torch = pytest.importorskip("torch")


def texture(h: int, w: int, seed: int = 0) -> np.ndarray:
    """Szum 1/f (jak naturalne zdjęcie) w skali szarości uint8."""
    rng = np.random.default_rng(seed)
    fy, fx = np.meshgrid(np.fft.fftfreq(h), np.fft.fftfreq(w), indexing="ij")
    spec = (rng.normal(size=(h, w)) + 1j * rng.normal(size=(h, w))) / np.maximum(np.hypot(fy, fx), 1 / max(h, w))
    img = np.real(np.fft.ifft2(spec))
    return ((img - img.min()) / np.ptp(img) * 255).astype(np.uint8)


def blur(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return cv2.filter2D(img, -1, cv2.flip(kernel, -1).astype(np.float32), borderType=cv2.BORDER_REPLICATE)


@pytest.mark.parametrize("params", deblur.DISK)
def test_dysk_jak_imagecorruptions(params):
    corruptions = pytest.importorskip("imagecorruptions.corruptions")
    assert np.allclose(deblur.disk_kernel(*params), corruptions.disk(*params))


@pytest.mark.parametrize("severity", range(5))
@pytest.mark.parametrize("angle", [-90.0, -33.3, 0.0, 17.0, 45.0])
def test_ruch_jak_imagecorruptions(severity, angle):
    corruptions = pytest.importorskip("imagecorruptions.corruptions")
    delta = np.zeros((81, 81), np.float32)
    delta[40, 40] = 1
    ref = deblur.trim(corruptions._motion_blur(delta, *deblur.MOTION[severity], angle).astype(np.float64))
    assert np.allclose(deblur.motion_kernel(*deblur.MOTION[severity], angle), ref, atol=1e-6)


def test_jadra_kwadratowe_nieparzyste_i_ruch_znormalizowany():
    kernels = deblur.candidate_kernels(12.0)
    assert len(kernels) == 15
    for k in kernels:
        assert k.shape[0] == k.shape[1] and k.shape[0] % 2 == 1 and 0.99 < k.sum() < 1.02
    assert all(k.sum() == pytest.approx(1) for k in kernels[5:])


def test_kandydaci_ruchu_w_obu_kierunkach():
    k = deblur.candidate_kernels(30.0)
    assert np.allclose(k[5], deblur.motion_kernel(*deblur.MOTION[0], 30.0))
    assert np.allclose(k[6], deblur.motion_kernel(*deblur.MOTION[0], -150.0))
    assert np.allclose(deblur.candidate_kernels(-40.0)[6], deblur.motion_kernel(*deblur.MOTION[0], 140.0))


def test_trim():
    k = np.zeros((9, 9))
    k[4, 4], k[4, 6] = 1, 1
    out = deblur.trim(k)
    assert out.shape == (5, 5) and out.sum() == pytest.approx(1) and out[2, 4] == pytest.approx(0.5)


def test_rl_jadro_jednostkowe_nie_zmienia_obrazu():
    img = np.dstack([texture(40, 60, s) for s in range(3)])
    (out,) = deblur.richardson_lucy(img, [np.ones((1, 1))], iters=5, device="cpu")
    assert np.abs(out.astype(int) - img).max() <= 1


def test_rl_wsadowo_jak_pojedynczo():
    img = np.dstack([texture(160, 256, s) for s in range(3)])
    kernels = [deblur.disk_kernel(*deblur.DISK[1]), deblur.motion_kernel(*deblur.MOTION[2], 25.0)]
    both = deblur.richardson_lucy(img, kernels, iters=10, device="cpu")
    m = max(k.shape[0] for k in kernels) // 2
    for k, out in zip(kernels, both, strict=True):
        (single,) = deblur.richardson_lucy(img, [k], iters=10, device="cpu")
        assert np.abs(out.astype(int) - single)[m:-m, m:-m].mean() < 0.5


@pytest.mark.parametrize(
    "kernel",
    [deblur.disk_kernel(*deblur.DISK[2]), deblur.motion_kernel(*deblur.MOTION[3], -20.0)],
    ids=["dysk", "ruch"],
)
def test_rl_przybliza_do_oryginalu(kernel):
    sharp = np.dstack([texture(256, 384, 7)] * 3)
    blurred = blur(sharp, kernel)
    (out,) = deblur.richardson_lucy(blurred, [kernel], iters=30, device="cpu")
    m = kernel.shape[0] // 2
    err_blur = np.abs(blurred.astype(int) - sharp)[m:-m, m:-m].mean()
    err_rl = np.abs(out.astype(int) - sharp)[m:-m, m:-m].mean()
    assert err_rl < 0.9 * err_blur


def test_deconv_views():
    crop = np.dstack([texture(30, 90, s) for s in range(3)])
    views = deblur.deconv_views(crop, 10.0, iters=2, device="cpu")
    assert len(views) == 16 and views[0] is crop
    assert all(v.shape == crop.shape and v.dtype == np.uint8 for v in views)


def test_estimacja_kata_ruchu():
    img = texture(1024, 1024, 3)
    blurred = blur(img, deblur.motion_kernel(*deblur.MOTION[3], 34.0))
    est = deblur.estimate_blur(blurred)
    assert est.kind == "motion_blur"
    assert abs(((est.motion_angle - 34.0) + 90) % 180 - 90) <= 4


def test_estymacja_nie_bierze_ostrego_ani_nieostrosci_za_ruch():
    img = texture(1024, 1024, 5)
    assert deblur.estimate_blur(img).kind != "motion_blur"
    assert deblur.estimate_blur(blur(img, deblur.disk_kernel(*deblur.DISK[3]))).kind != "motion_blur"


def test_widmo_malego_obrazu():
    assert deblur.image_log_spectrum(texture(100, 300)).shape == (24, 36)


@pytest.mark.parametrize(("corruption", "severity"), [("motion_blur", 2), ("motion_blur", 5), ("defocus_blur", 3)])
def test_prawdziwe_jadro_jak_w_budowie_zbioru(corruption, severity):
    pytest.importorskip("imagecorruptions")
    from alpr.build.corruptions import apply_corruption

    delta = np.zeros((101, 101, 3), np.uint8)
    delta[50, 50] = 255
    key = f"zdjecie.jpg:{corruption}_{severity}"
    response = apply_corruption(delta, corruption, severity, key)[:, :, 0].astype(np.float64) / 255
    kernel = deblur.true_kernel(corruption, severity, key)
    r = kernel.shape[0] // 2
    assert np.abs(response[50 - r : 51 + r, 50 - r : 51 + r] - kernel).max() < 0.01


def test_prawdziwe_jadro_brak_dla_innych():
    assert deblur.true_kernel("darken", 3, "x") is None


def test_deconv_training_crops(tmp_path):
    (tmp_path / "images" / "defocus_blur_2").mkdir(parents=True)
    cv2.imwrite(str(tmp_path / "images/defocus_blur_2/a.jpg"), np.dstack([texture(120, 200, 1)] * 3))
    rows = [
        {"variant": v, "artificial_corruption": c, "artificial_severity": s, "image": f"images/{v}/a.jpg", "split": sp}
        | {"source_image": "x/a.jpg", "text": "AB1", "x1": 40, "y1": 40, "x2": 160, "y2": 80}
        for v, c, s, sp in [("defocus_blur_2", "defocus_blur", 2, "train"), ("darken_1", "darken", 1, "train")]
        + [("defocus_blur_2", "defocus_blur", 2, "val")]
    ]
    pd.DataFrame(rows).to_csv(tmp_path / "ground_truth.csv", index=False)
    crops, texts = deblur.deconv_training_crops(tmp_path, "train", iters=2, copies=3)
    assert texts == ["AB1"] * 3 and all(c.dtype == np.uint8 and c.shape[2] == 3 for c in crops)


def test_estymacja_wielu_plikow(tmp_path):
    paths = []
    for i, angle in enumerate([10.0, -60.0]):
        img = blur(texture(600, 600, i), deblur.motion_kernel(*deblur.MOTION[3], angle))
        paths.append(tmp_path / f"{i}.png")
        cv2.imwrite(str(paths[-1]), img)
    one = deblur.estimate_blur_files(paths, workers=1)
    assert one == deblur.estimate_blur_files(paths, workers=2)
    assert all(e.kind == "motion_blur" for e in one)
