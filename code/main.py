import numpy as np
import tensorflow as tf
import math

[(337, 0, 728, 995),
 (337, 0, 728, 183),
 (337, 0, 728, 866),
 (337, 0, 728, 153),
 (337, 0, 728, 414),
 (497, 0, 848, 512),
 (497, 0, 848, 933),
 (497, 0, 848, 608),
 (497, 0, 848, 180),
 (497, 0, 848, 594),
 (39, 2, 8, 219),
 (39, 2, 8, 946),
 (39, 2, 8, 185),
 (39, 2, 8, 630),
 (39, 2, 8, 669),
 (101, 2, 14, 882),
 (101, 2, 14, 745),
 (101, 2, 14, 687),
 (101, 2, 14, 457),
 (101, 2, 14, 253),
 (293, 0, 342, 824),
 (293, 0, 342, 433),
 (293, 0, 342, 576),
 (293, 0, 342, 12),
 (293, 0, 342, 785),
 (546, 5, 681, 870),
 (546, 5, 681, 189),
 (546, 5, 681, 635),
 (546, 5, 681, 829),
 (546, 5, 681, 414),
 (657, 7, 359, 547),
 (657, 7, 359, 311),
 (657, 7, 359, 450),
 (657, 7, 359, 101),
 (657, 7, 359, 353),
 (449, 2, 204, 397),
 (449, 2, 204, 630),
 (449, 2, 204, 6),
 (449, 2, 204, 338),
 (449, 2, 204, 441),
 (949, 1, 683, 693),
 (949, 1, 683, 578),
 (949, 1, 683, 81),
 (949, 1, 683, 253),
 (949, 1, 683, 161),
 (271, 2, 443, 381),
 (271, 2, 443, 651),
 (271, 2, 443, 738),
 (271, 2, 443, 15),
 (271, 2, 443, 65),
 (509, 1, 344, 148),
 (509, 1, 344, 786),
 (509, 1, 344, 285),
 (509, 1, 344, 939),
 (509, 1, 344, 395),
 (78, 2, 51, 966),
 (78, 2, 51, 580),
 (78, 2, 51, 256),
 (78, 2, 51, 664),
 (78, 2, 51, 414)]

def main():
    { < tf.Tensor
    'Placeholder:0'
    shape = (1,)
    dtype = bool >: [False], < tf.Tensor
    'batch_0:0'
    shape = (?, 3)
    dtype = int32 >: [(337, 728, 995), (337, 728, 183), (337, 728, 866), (337, 728, 153), (337, 728, 414),
                      (497, 848, 512), (497, 848, 933), (497, 848, 608), (497, 848, 180), (497, 848, 594),
                      (293, 342, 824), (293, 342, 433), (293, 342, 576), (293, 342, 12), (293, 342, 785)], < tf.Tensor
    'label_0:0'
    shape = (?, 1)
    dtype = float32 >: [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
                        [0.0], [0.0]], < tf.Tensor
    'batch_1:0'
    shape = (?, 3)
    dtype = int32 >: [(949, 683, 693), (949, 683, 578), (949, 683, 81), (949, 683, 253), (949, 683, 161),
                      (509, 344, 148), (509, 344, 786), (509, 344, 285), (509, 344, 939), (509, 344, 395)], < tf.Tensor
    'label_1:0'
    shape = (?, 1)
    dtype = float32 >: [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], < tf.Tensor
    'batch_2:0'
    shape = (?, 3)
    dtype = int32 >: [(39, 8, 219), (39, 8, 946), (39, 8, 185), (39, 8, 630), (39, 8, 669), (101, 14, 882),
                      (101, 14, 745), (101, 14, 687), (101, 14, 457), (101, 14, 253), (449, 204, 397), (449, 204, 630),
                      (449, 204, 6), (449, 204, 338), (449, 204, 441), (271, 443, 381), (271, 443, 651),
                      (271, 443, 738), (271, 443, 15), (271, 443, 65), (78, 51, 966), (78, 51, 580), (78, 51, 256),
                      (78, 51, 664), (78, 51, 414)], < tf.Tensor
    'label_2:0'
    shape = (?, 1)
    dtype = float32 >: [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
                        [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], < tf.Tensor
    'batch_3:0'
    shape = (?, 3)
    dtype = int32 >: [], < tf.Tensor
    'label_3:0'
    shape = (?, 1)
    dtype = float32 >: [], < tf.Tensor
    'batch_4:0'
    shape = (?, 3)
    dtype = int32 >: [], < tf.Tensor
    'label_4:0'
    shape = (?, 1)
    dtype = float32 >: [], < tf.Tensor
    'batch_5:0'
    shape = (?, 3)
    dtype = int32 >: [(546, 681, 870), (546, 681, 189), (546, 681, 635), (546, 681, 829), (546, 681, 414)], < tf.Tensor
    'label_5:0'
    shape = (?, 1)
    dtype = float32 >: [[0.0], [0.0], [0.0], [0.0], [0.0]], < tf.Tensor
    'batch_6:0'
    shape = (?, 3)
    dtype = int32 >: [], < tf.Tensor
    'label_6:0'
    shape = (?, 1)
    dtype = float32 >: [], < tf.Tensor
    'batch_7:0'
    shape = (?, 3)
    dtype = int32 >: [(657, 359, 547), (657, 359, 311), (657, 359, 450), (657, 359, 101), (657, 359, 353)], < tf.Tensor
    'label_7:0'
    shape = (?, 1)
    dtype = float32 >: [[0.0], [0.0], [0.0], [0.0], [0.0]], < tf.Tensor
    'batch_8:0'
    shape = (?, 3)
    dtype = int32 >: [], < tf.Tensor
    'label_8:0'
    shape = (?, 1)
    dtype = float32 >: []}


if __name__ == '__main__':
    main()
