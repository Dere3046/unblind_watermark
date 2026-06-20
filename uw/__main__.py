import argparse
import cv2
from .core import erase


def main():
    ap = argparse.ArgumentParser(
        description='erase blind watermark via DWT-DCT-SVD overwrite'
    )
    ap.add_argument('input', help='input image path')
    ap.add_argument('output', help='output image path')
    ap.add_argument('--d1', type=float, default=36,
                    help='quantization step for S[0]')
    ap.add_argument('--d2', type=float, default=20,
                    help='quantization step for S[1]')
    args = ap.parse_args()

    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f'cannot read {args.input}')

    result = erase(img, d1=args.d1, d2=args.d2)
    cv2.imwrite(args.output, result)
    print(f'saved {args.output}')


if __name__ == '__main__':
    main()
