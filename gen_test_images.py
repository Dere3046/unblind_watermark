import numpy as np
import cv2


def _rng():
    return np.random.RandomState(0)


def _save(name, img, size=(256, 256)):
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.shape[:2] != size:
        img = cv2.resize(img, size[::-1])
    cv2.imwrite(f'test_images/{name}.png', img)
    print(f'  {name}.png  {img.shape}')


def generate():
    size = (256, 256)
    h, w = size

    rng = _rng()

    img = np.full((h, w, 3), 128, dtype=np.uint8)
    _save('test_uniform', img)

    grad = np.linspace(0, 255, w, dtype=np.float32)
    img = np.tile(grad, (h, 1))
    img = np.stack([img, img * 0.7 + 50, img * 0.3 + 30], axis=2)
    _save('test_gradient_h', img)

    grad = np.linspace(0, 255, h, dtype=np.float32)
    img = np.tile(grad[:, None], (1, w))
    img = np.stack([img * 0.5 + 20, img * 0.8 + 10, img * 0.6 + 40], axis=2)
    _save('test_gradient_v', img)

    noise = rng.randn(h, w, 3).astype(np.float32) * 40 + 128
    _save('test_noise_gaussian', noise)

    img = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(0, h, 16):
        for j in range(0, w, 16):
            if (i // 16 + j // 16) % 2 == 0:
                img[i:i + 16, j:j + 16] = 200
    _save('test_checkerboard', img)

    img = np.zeros((h, w, 3), dtype=np.uint8)
    for _ in range(30):
        cx, cy = rng.randint(0, w), rng.randint(0, h)
        r = rng.randint(10, 50)
        color = rng.randint(0, 255, 3).tolist()
        cv2.circle(img, (cx, cy), r, color, -1)
    for _ in range(15):
        x1, y1 = rng.randint(0, w - 40), rng.randint(0, h - 40)
        color = rng.randint(0, 255, 3).tolist()
        cv2.rectangle(img, (x1, y1), (x1 + rng.randint(10, 40),
                                      y1 + rng.randint(10, 40)), color, -1)
    _save('test_shapes', img)

    noise = rng.randn(h, w).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (25, 25), 8)
    noise = (noise - noise.min()) / (noise.max() - noise.min()) * 255
    img = np.stack([noise, noise * 0.8, noise * 0.6], axis=2)
    _save('test_texture', img)

    yy, xx = np.mgrid[:h, :w]
    cx, cy = h // 2, w // 2
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    dist = dist / dist.max()
    r = np.clip(dist * 255, 0, 255).astype(np.float32)
    g = np.clip((1 - dist) * 255, 0, 255).astype(np.float32)
    b = np.clip(np.sin(dist * np.pi) * 255, 0, 255).astype(np.float32)
    img = np.stack([b, g, r], axis=2)
    noise_small = rng.randn(h // 4, w // 4).astype(np.float32) * 20
    noise_small = cv2.resize(noise_small, (w, h))
    img = np.clip(img + noise_small[..., None], 0, 255)
    _save('test_radial', img)

    img = np.zeros((h, w, 3), dtype=np.uint8)
    for _ in range(50):
        x, y = rng.randint(0, w - 1), rng.randint(0, h - 1)
        c = rng.randint(30, 220)
        cv2.rectangle(img, (x, y), (x + rng.randint(5, 25),
                                    y + rng.randint(5, 25)),
                      (c, c // 2 + 60, 255 - c), -1)
    _save('test_squares', img)

    print(f'\n9 images generated in test_images/')


if __name__ == '__main__':
    generate()
