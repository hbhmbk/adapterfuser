import math
import logging
logger = logging.getLogger(__name__)
import sys
def setup_logging(args=None, log_level=None, reset=False):
    if logging.root.handlers:
        if reset:
            # remove all handlers
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
        else:
            return

    # log_level can be set by the caller or by the args, the caller has priority. If not set, use INFO
    if log_level is None and args is not None:
        log_level = args.console_log_level
    if log_level is None:
        log_level = "INFO"
    log_level = getattr(logging, log_level)

    msg_init = None
    if args is not None and args.console_log_file:
        handler = logging.FileHandler(args.console_log_file, mode="w")
    else:
        handler = None
        if not args or not args.console_log_simple:
            try:
                from rich.logging import RichHandler
                from rich.console import Console
                from rich.logging import RichHandler

                handler = RichHandler(console=Console(stderr=True))
            except ImportError:
                # print("rich is not installed, using basic logging")
                msg_init = "rich is not installed, using basic logging"

        if handler is None:
            handler = logging.StreamHandler(sys.stdout)  # same as print
            handler.propagate = False

    formatter = logging.Formatter(
        fmt="%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logging.root.setLevel(log_level)
    logging.root.addHandler(handler)

    if msg_init is not None:
        logger = logging.getLogger(__name__)
        logger.info(msg_init)

def make_bucket_resolutions(max_reso, min_size=256, max_size=1024, divisible=64):
    max_width, max_height = max_reso
    max_area = max_width * max_height

    resos = set()

    width = int(math.sqrt(max_area) // divisible) * divisible
    resos.add((width, width))

    width = min_size
    while width <= max_size:
        height = min(max_size, int((max_area // width) // divisible) * divisible)
        if height >= min_size:
            resos.add((width, height))
            resos.add((height, width))

        # # make additional resos
        # if width >= height and width - divisible >= min_size:
        #   resos.add((width - divisible, height))
        #   resos.add((height, width - divisible))
        # if height >= width and height - divisible >= min_size:
        #   resos.add((width, height - divisible))
        #   resos.add((height - divisible, width))

        width += divisible

    resos = list(resos)
    resos.sort()
    return resos



if __name__ == "__main__":
    resos = make_bucket_resolutions((512, 768))

    logger.info(f"{len(resos)}")
    logger.info(f"{resos}")
    aspect_ratios = [w / h for w, h in resos]
    logger.info(f"{aspect_ratios}")

    ars = set()
    for ar in aspect_ratios:
        if ar in ars:
            logger.error(f"error! duplicate ar: {ar}")
        ars.add(ar)

# class BucketManager:
#     def __init__(self, no_upscale, max_reso, min_size, max_size, reso_steps) -> None:
#         if max_size is not None:
#             if max_reso is not None:
#                 assert max_size >= max_reso[0], "the max_size should be larger than the width of max_reso"
#                 assert max_size >= max_reso[1], "the max_size should be larger than the height of max_reso"
#             if min_size is not None:
#                 assert max_size >= min_size, "the max_size should be larger than the min_size"

#         self.no_upscale = no_upscale
#         if max_reso is None:
#             self.max_reso = None
#             self.max_area = None
#         else:
#             self.max_reso = max_reso
#             self.max_area = max_reso[0] * max_reso[1]
#         self.min_size = min_size
#         self.max_size = max_size
#         self.reso_steps = reso_steps

#         self.resos = []
#         self.reso_to_id = {}
#         self.buckets = []  # 前処理時は (image_key, image, original size, crop left/top)、学習時は image_key

#     def add_image(self, reso, image_or_info):
#         bucket_id = self.reso_to_id[reso]
#         self.buckets[bucket_id].append(image_or_info)
#         print(self.buckets)

#     def shuffle(self):
#         for bucket in self.buckets:
#             random.shuffle(bucket)

#     def sort(self):
#         # 解像度順にソートする（表示時、メタデータ格納時の見栄えをよくするためだけ）。bucketsも入れ替えてreso_to_idも振り直す
#         sorted_resos = self.resos.copy()
#         sorted_resos.sort()

#         sorted_buckets = []
#         sorted_reso_to_id = {}
#         for i, reso in enumerate(sorted_resos):
#             bucket_id = self.reso_to_id[reso]
#             sorted_buckets.append(self.buckets[bucket_id])
#             sorted_reso_to_id[reso] = i

#         self.resos = sorted_resos
#         self.buckets = sorted_buckets
#         self.reso_to_id = sorted_reso_to_id

#     def make_buckets(self):
#         resos = make_bucket_resolutions(self.max_reso, self.min_size, self.max_size, self.reso_steps)
#         self.set_predefined_resos(resos)

#     def set_predefined_resos(self, resos):
#         # 規定サイズから選ぶ場合の解像度、aspect ratioの情報を格納しておく
#         self.predefined_resos = resos.copy()
#         self.predefined_resos_set = set(resos)
#         self.predefined_aspect_ratios = np.array([w / h for w, h in resos])

#     def add_if_new_reso(self, reso):
#         if reso not in self.reso_to_id:
#             bucket_id = len(self.resos)
#             self.reso_to_id[reso] = bucket_id
#             self.resos.append(reso)
#             self.buckets.append([])
#             print(self.resos)
#             # logger.info(reso, bucket_id, len(self.buckets))

#     def round_to_steps(self, x):
#         x = int(x + 0.5)
#         return x - x % self.reso_steps

#     def select_bucket(self, image_width, image_height):
#         aspect_ratio = image_width / image_height
#         if not self.no_upscale:
#             # 拡大および縮小を行う
#             # 同じaspect ratioがあるかもしれないので（fine tuningで、no_upscale=Trueで前処理した場合）、解像度が同じものを優先する
#             reso = (image_width, image_height)
#             if reso in self.predefined_resos_set:
#                 pass
#             else:
#                 ar_errors = self.predefined_aspect_ratios - aspect_ratio
#                 predefined_bucket_id = np.abs(ar_errors).argmin()  # 当該解像度以外でaspect ratio errorが最も少ないもの
#                 reso = self.predefined_resos[predefined_bucket_id]

#             ar_reso = reso[0] / reso[1]
#             if aspect_ratio > ar_reso:  # 横が長い→縦を合わせる
#                 scale = reso[1] / image_height
#             else:
#                 scale = reso[0] / image_width

#             resized_size = (int(image_width * scale + 0.5), int(image_height * scale + 0.5))
#             # logger.info(f"use predef, {image_width}, {image_height}, {reso}, {resized_size}")
#         else:
#             # 縮小のみを行う
#             if image_width * image_height > self.max_area:
#                 # 画像が大きすぎるのでアスペクト比を保ったまま縮小することを前提にbucketを決める
#                 resized_width = math.sqrt(self.max_area * aspect_ratio)
#                 resized_height = self.max_area / resized_width
#                 assert abs(resized_width / resized_height - aspect_ratio) < 1e-2, "aspect is illegal"

#                 # リサイズ後の短辺または長辺をreso_steps単位にする：aspect ratioの差が少ないほうを選ぶ
#                 # 元のbucketingと同じロジック
#                 b_width_rounded = self.round_to_steps(resized_width)
#                 b_height_in_wr = self.round_to_steps(b_width_rounded / aspect_ratio)
#                 ar_width_rounded = b_width_rounded / b_height_in_wr

#                 b_height_rounded = self.round_to_steps(resized_height)
#                 b_width_in_hr = self.round_to_steps(b_height_rounded * aspect_ratio)
#                 ar_height_rounded = b_width_in_hr / b_height_rounded

#                 # logger.info(b_width_rounded, b_height_in_wr, ar_width_rounded)
#                 # logger.info(b_width_in_hr, b_height_rounded, ar_height_rounded)

#                 if abs(ar_width_rounded - aspect_ratio) < abs(ar_height_rounded - aspect_ratio):
#                     resized_size = (b_width_rounded, int(b_width_rounded / aspect_ratio + 0.5))
#                 else:
#                     resized_size = (int(b_height_rounded * aspect_ratio + 0.5), b_height_rounded)
#                 # logger.info(resized_size)
#             else:
#                 resized_size = (image_width, image_height)  # リサイズは不要

#             # 画像のサイズ未満をbucketのサイズとする（paddingせずにcroppingする）
#             bucket_width = resized_size[0] - resized_size[0] % self.reso_steps
#             bucket_height = resized_size[1] - resized_size[1] % self.reso_steps
#             # logger.info(f"use arbitrary {image_width}, {image_height}, {resized_size}, {bucket_width}, {bucket_height}")

#             reso = (bucket_width, bucket_height)

#         self.add_if_new_reso(reso)

#         ar_error = (reso[0] / reso[1]) - aspect_ratio
#         return reso, resized_size, ar_error

#     @staticmethod
#     def get_crop_ltrb(bucket_reso: Tuple[int, int], image_size: Tuple[int, int]):
#         # Stability AIの前処理に合わせてcrop left/topを計算する。crop rightはflipのaugmentationのために求める
#         # Calculate crop left/top according to the preprocessing of Stability AI. Crop right is calculated for flip augmentation.

#         bucket_ar = bucket_reso[0] / bucket_reso[1]
#         image_ar = image_size[0] / image_size[1]
#         if bucket_ar > image_ar:
#             # bucketのほうが横長→縦を合わせる
#             resized_width = bucket_reso[1] * image_ar
#             resized_height = bucket_reso[1]
#         else:
#             resized_width = bucket_reso[0]
#             resized_height = bucket_reso[0] / image_ar
#         crop_left = (bucket_reso[0] - resized_width) // 2
#         crop_top = (bucket_reso[1] - resized_height) // 2
#         crop_right = crop_left + resized_width
#         crop_bottom = crop_top + resized_height
#         return crop_left, crop_top, crop_right, crop_bottom