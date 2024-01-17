from einops import rearrange

import tensorflow as tf
from tensorflow.keras import Sequential
import tensorflow.keras.initializers as tfki
import tensorflow.keras.layers as tfkl
import tensorflow.keras.models as tfkm


def dim_swap(tensor, order=[]):
    if len(order) != 2:
        raise ValueError("Order list must have exactly two elements.")

    ndims = len(tensor.shape)

    order = [d if d >= 0 else ndims + d for d in order]

    if order[0] >= ndims or order[1] >= ndims or order[0] < 0 or order[1] < 0:
        raise IndexError("Order indices are out of range for the tensor dimensions.")

    perm = list(range(ndims))
    perm[order[0]], perm[order[1]] = perm[order[1]], perm[order[0]]

    return tf.transpose(tensor, perm=perm)


def flatten(input_tensor, start_dim, end_dim):
    shape = tf.shape(input_tensor)
    slice_numel = tf.reduce_prod(shape[start_dim : end_dim + 1])
    new_shape = tf.concat(
        [
            shape[:start_dim],
            [slice_numel],
            shape[end_dim + 1 :],
        ],
        axis=0,
    )

    return tf.reshape(input_tensor, new_shape)


class PreNorm(tfkl.Layer):
    def __init__(self, fn):
        super(PreNorm, self).__init__()
        self.fn = fn
        self.norm = tfkl.LayerNormalization(axis=-1, epsilon=1e-6)

    def call(self, x):
        x = self.norm(x)
        return self.fn(x)


class Illumination_Estimator(tfkl.Layer):
    def __init__(self, n_fea_middle, n_fea_out=3):
        super(Illumination_Estimator, self).__init__()

        self.conv1 = tfkl.Conv2D(n_fea_middle, kernel_size=1, use_bias=True)
        self.depth_conv = tfkl.DepthwiseConv2D(
            kernel_size=5, padding="same", use_bias=True
        )

        self.conv2 = tfkl.Conv2D(n_fea_out, kernel_size=1, use_bias=True)

    def call(self, img):
        mean_c = tf.expand_dims(tf.reduce_mean(img, axis=3), axis=3)
        input = tf.concat([img, mean_c], axis=3)

        x_1 = self.conv1(input)
        illu_fea = self.depth_conv(x_1)
        illu_map = self.conv2(illu_fea)
        return illu_fea, illu_map


class IG_MSA(tfkm.Model):
    def __init__(
        self,
        dim,
        dim_head=40,
        heads=8,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = tfkl.Dense(
            dim_head * heads,
            use_bias=False,
            kernel_initializer=tfki.TruncatedNormal(stddev=0.02),
        )
        self.to_k = tfkl.Dense(
            dim_head * heads,
            use_bias=False,
            kernel_initializer=tfki.TruncatedNormal(stddev=0.02),
        )
        self.to_v = tfkl.Dense(
            dim_head * heads,
            use_bias=False,
            kernel_initializer=tfki.TruncatedNormal(stddev=0.02),
        )
        self.rescale = tf.Variable(tf.ones([heads, 1, 1]))
        self.proj = tfkl.Dense(
            dim, use_bias=True, kernel_initializer=tfki.TruncatedNormal(stddev=0.02)
        )
        self.pos_emb = Sequential(
            [
                tfkl.DepthwiseConv2D(
                    kernel_size=3,
                    strides=1,
                    padding="same",
                    use_bias=True,
                    activation="gelu",
                ),
                tfkl.DepthwiseConv2D(kernel_size=3, padding="same", use_bias=True),
            ]
        )

        self.dim = dim

    def call(self, x_in, illu_fea_trans):
        b, h, w, c = (
            tf.shape(x_in)[0],
            tf.shape(x_in)[1],
            tf.shape(x_in)[2],
            tf.shape(x_in)[3],
        )
        x = tf.reshape(x_in, [b, h * w, c])
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        illu_attn = illu_fea_trans
        illu_attn_flat = flatten(illu_attn, 1, 2)
        q, k, v, illu_attn = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads),
            (q_inp, k_inp, v_inp, illu_attn_flat),
        )
        v = v * illu_attn
        q = dim_swap(q, [-2, -1])
        k = dim_swap(k, [-2, -1])
        v = dim_swap(v, [-2, -1])
        q = tf.nn.l2_normalize(q)
        k = tf.nn.l2_normalize(k)
        attn = tf.matmul(k, dim_swap(q, [-2, -1]))
        attn = attn * self.rescale
        attn = tf.nn.softmax(attn)
        x = tf.matmul(attn, v)
        x = tf.transpose(x, [0, 3, 1, 2])
        x = tf.reshape(x, [b, h * w, self.num_heads * self.dim_head])
        out_c = tf.reshape(self.proj(x), [b, h, w, c])
        out_p = self.pos_emb(tf.reshape(v_inp, [b, h, w, c]))
        out = out_c + out_p

        return out


class FeedForward(tfkm.Model):
    def __init__(self):
        super().__init__()
        self.net = Sequential(
            [
                tfkl.DepthwiseConv2D(
                    kernel_size=1,
                    strides=1,
                    use_bias=False,
                    activation="gelu",
                ),
                tfkl.DepthwiseConv2D(
                    kernel_size=3,
                    strides=1,
                    padding="same",
                    use_bias=False,
                    activation="gelu",
                ),
                tfkl.DepthwiseConv2D(kernel_size=1, strides=1, use_bias=False),
            ]
        )

    def call(self, x):
        out = self.net(x)
        return out


class IGAB(tfkl.Layer):
    def __init__(self, dim, dim_head=40, heads=8, num_blocks=2):
        super().__init__()
        self.blocks = []
        for _ in range(num_blocks):
            self.blocks.append(
                [
                    IG_MSA(dim=dim, dim_head=dim_head, heads=heads),
                    PreNorm(fn=FeedForward()),
                ]
            )

    def call(self, x, illu_fea):
        for attn, ff in self.blocks:
            x = attn(x, illu_fea_trans=illu_fea) + x
            x = ff(x) + x
        out = x
        return out


class Corruption_Restorer(tfkl.Layer):
    def __init__(self, out_dim=3, dim=40, level=2, num_blocks=[1, 2, 2]):
        super(Corruption_Restorer, self).__init__()
        self.dim = dim
        self.level = level

        # Input projection
        self.embedding = tfkl.Conv2D(
            self.dim, kernel_size=3, strides=1, padding="same", use_bias=False
        )

        # Encoder
        self.encoder_layers = []
        dim_level = dim
        for i in range(level):
            self.encoder_layers.append(
                [
                    IGAB(
                        dim=dim_level,
                        num_blocks=num_blocks[i],
                        dim_head=dim,
                        heads=dim_level // dim,
                    ),
                    tfkl.Conv2D(
                        dim_level * 2,
                        kernel_size=4,
                        strides=2,
                        padding="same",
                        use_bias=False,
                    ),
                    tfkl.Conv2D(
                        dim_level * 2,
                        kernel_size=4,
                        strides=2,
                        padding="same",
                        use_bias=False,
                    ),
                ]
            )
            dim_level *= 2

        # Bottleneck
        self.bottleneck = IGAB(
            dim=dim_level,
            dim_head=dim,
            heads=dim_level // dim,
            num_blocks=num_blocks[-1],
        )

        # Decoder
        self.decoder_layers = []
        for i in range(level):
            self.decoder_layers.append(
                [
                    tfkl.Conv2DTranspose(dim_level // 2, kernel_size=2, strides=2),
                    tfkl.Conv2D(
                        dim_level // 2, kernel_size=1, strides=1, use_bias=False
                    ),
                    IGAB(
                        dim=dim_level // 2,
                        num_blocks=num_blocks[level - 1 - i],
                        dim_head=dim,
                        heads=(dim_level // 2) // dim,
                    ),
                ]
            )
            dim_level //= 2

        # Output projection
        self.mapping = tfkl.Conv2D(
            out_dim, kernel_size=3, strides=1, padding="same", use_bias=False
        )

    def call(self, x, illu_fea):
        # Embedding
        fea = self.embedding(x)

        # Encoder
        fea_encoder = []
        illu_fea_list = []
        for IGAB, FeaDownSample, IlluFeaDownsample in self.encoder_layers:
            fea = IGAB(fea, illu_fea)
            illu_fea_list.append(illu_fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)
            illu_fea = IlluFeaDownsample(illu_fea)

        # Bottleneck
        fea = self.bottleneck(fea, illu_fea)

        # Decoder
        for i, (FeaUpSample, Fution, LeWinBlcok) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(
                tfkl.concatenate([fea, fea_encoder[self.level - 1 - i]], axis=-1)
            )
            illu_fea = illu_fea_list[self.level - 1 - i]
            fea = LeWinBlcok(fea, illu_fea)

        # Mapping
        out = self.mapping(fea) + x
        return out


class RetinexFormer_Single_Stage(tfkl.Layer):
    def __init__(self, out_channels=3, n_feat=40, level=2, num_blocks=[1, 2, 2]):
        super(RetinexFormer_Single_Stage, self).__init__()
        self.estimator = Illumination_Estimator(n_feat)
        self.denoiser = Corruption_Restorer(
            out_dim=out_channels,
            dim=n_feat,
            level=level,
            num_blocks=num_blocks,
        )

    def call(self, img):
        illu_fea, illu_map = self.estimator(img)
        input_img = img * illu_map + img
        output_img = self.denoiser(input_img, illu_fea)

        return output_img


class RetinexFormer(tfkm.Model):
    def __init__(self, out_channels=3, n_feat=40, stage=1, num_blocks=[1, 2, 2]):
        super(RetinexFormer, self).__init__()
        self.stage = stage

        self.body = Sequential(
            [
                RetinexFormer_Single_Stage(
                    out_channels=out_channels,
                    n_feat=n_feat,
                    level=2,
                    num_blocks=num_blocks,
                )
                for _ in range(stage)
            ]
        )

    def call(self, x):
        out = self.body(x)
        return out
