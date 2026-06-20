import pytest
import numpy as np
import cv2
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'blind_watermark'))

from uw import erase
from blind_watermark import WaterMark

TEST_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'test_images')
PW_IMG = 1
PW_WM = 1


def _bit_len(text):
    return len(bin(int(text.encode('utf-8').hex(), base=16))[2:])


def _get_images():
    files = sorted(os.listdir(TEST_DIR))
    return [f for f in files if f.endswith('.png')]


@pytest.fixture(scope='module')
def test_images():
    files = _get_images()
    assert len(files) >= 9, f'need at least 9 test images, found {len(files)}'
    images = {}
    for f in files:
        path = os.path.join(TEST_DIR, f)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        assert img is not None, f'cannot read {path}'
        images[f.replace('.png', '')] = img
    return images


@pytest.fixture(scope='module')
def watermarked_images(test_images):
    wm_text = 'UNBLIND_TEST_WATERMARK'
    wm_bits = _bit_len(wm_text)
    result = {}
    for name, img in test_images.items():
        bwm = WaterMark(password_img=PW_IMG, password_wm=PW_WM)
        bwm.read_img(img=img)
        bwm.read_wm(wm_text, mode='str')
        emb = bwm.embed()
        extracted = WaterMark(password_img=PW_IMG, password_wm=PW_WM).extract(
            embed_img=emb, wm_shape=wm_bits, mode='str'
        )
        assert extracted == wm_text, \
            f'{name}: embed verify failed: {repr(extracted)} != {repr(wm_text)}'
        result[name] = emb
    return result, wm_text, wm_bits


@pytest.fixture(scope='module')
def erased_images(watermarked_images):
    wm_imgs, wm_text, wm_bits = watermarked_images
    result = {}
    for name, img in wm_imgs.items():
        erased = erase(img)
        result[name] = erased
    return result, wm_text, wm_bits


class TestEraseBlindWatermark:

    def test_images_count(self, test_images):
        assert len(test_images) >= 9

    def test_watermark_destroyed(self, erased_images, watermarked_images):
        erased, wm_text, wm_bits = erased_images
        failures = []
        for name in erased:
            bwm = WaterMark(password_img=PW_IMG, password_wm=PW_WM)
            try:
                ext = bwm.extract(embed_img=erased[name],
                                  wm_shape=wm_bits, mode='str')
                if ext == wm_text:
                    failures.append((name, ext))
            except Exception:
                pass
        assert not failures, \
            f'watermark still intact on: {failures}'

    def test_visual_quality(self, erased_images, watermarked_images):
        erased, _, _ = erased_images
        wm_imgs, _, _ = watermarked_images
        failures = []
        for name in erased:
            ref = np.clip(wm_imgs[name], 0, 255).astype(np.uint8)
            psnr = cv2.PSNR(ref, erased[name])
            if psnr < 30:
                failures.append((name, psnr))
        assert not failures, \
            f'low PSNR: {failures}'

    def test_extraction_garbled(self, erased_images, watermarked_images):
        erased, wm_text, wm_bits = erased_images
        failures = []
        for name in erased:
            bwm = WaterMark(password_img=PW_IMG, password_wm=PW_WM)
            try:
                ext = bwm.extract(embed_img=erased[name],
                                  wm_shape=wm_bits, mode='str')
                common = set(wm_text) & set(ext)
                if len(common) > len(wm_text) * 0.7:
                    failures.append((name, ext, common))
            except Exception:
                pass
        assert not failures, \
            f'too many chars survived: {failures}'

    def test_brute_force_no_recovery(self, erased_images, watermarked_images):
        erased, wm_text, wm_bits = erased_images
        for name in list(erased.keys())[:3]:
            for pw in [1, 2, 3, 42, 100, 999]:
                try:
                    bwm = WaterMark(password_img=pw, password_wm=pw)
                    ext = bwm.extract(embed_img=erased[name],
                                      wm_shape=wm_bits, mode='str')
                    if ext == wm_text:
                        pytest.fail(
                            f'{name}: watermark recovered with pw_img={pw}')
                except Exception:
                    pass
