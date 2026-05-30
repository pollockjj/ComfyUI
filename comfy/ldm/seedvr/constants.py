"""SeedVR2 constants, grouped by provenance (BYTEDANCE_* / SEEDVR2_* / standards)."""

# Progressive-sampler chunk-size law: frames = FRAMES_PER_GB * (free_GB - GB_MARGIN), 4n+1.
SEEDVR2_CHUNK_GB_MARGIN = 3
SEEDVR2_CHUNK_FRAMES_PER_GB = 4

SEEDVR2_7B_VID_DIM = 3072              # 3b-vs-7b sentinel (3072 is ByteDance 7b vid_dim)
SEEDVR2_OOM_BACKOFF_DIVISOR = 2
SEEDVR2_DTYPE_BYTES_FLOOR = 4
SEEDVR2_7B_MLP_CHUNK = 8192
SEEDVR2_ROPE_PARTIAL_CHUNK_TOKENS = 4096
SEEDVR2_LATENT_CHANNELS = 16
SEEDVR2_COND_CHANNELS = 17             # vid_in_channels(33) - latent(16)
SEEDVR2_DEFAULT_TEMPORAL_SIZE = 16
SEEDVR2_COLOR_MEM_HEADROOM = 0.75
SEEDVR2_LAB_SCALE_MULTIPLIER = 13
SEEDVR2_WAVELET_SCALE_MULTIPLIER = 10
SEEDVR2_ADAIN_SCALE_MULTIPLIER = 6

BYTEDANCE_VAE_SCALING_FACTOR = 0.9152  # configs_3b/main.yaml:57
BYTEDANCE_VAE_SHIFTING_FACTOR = 0.0    # infer.py shifting_factor
BYTEDANCE_VAE_CONV_MEM_GIB = 0.5       # configs_3b/main.yaml:54
BYTEDANCE_VAE_NORM_MEM_GIB = 0.5       # configs_3b/main.yaml:55
BYTEDANCE_LOGVAR_CLAMP_MIN = -30.0     # video_vae_v3/modules/types.py:28
BYTEDANCE_LOGVAR_CLAMP_MAX = 20.0      # video_vae_v3/modules/types.py:28
BYTEDANCE_GN_CHUNKS_FP16 = 4           # causal_inflation_lib.py:351
BYTEDANCE_GN_CHUNKS_FP32 = 2           # causal_inflation_lib.py:351
BYTEDANCE_CONTIGUOUS_BATCH_THRESHOLD = 64  # attn_video_vae.py:308
BYTEDANCE_BLOCK_OUT_CHANNELS = (128, 256, 512, 512)  # s8_c16_t4_inflation_sd3.yaml:7-11
BYTEDANCE_SLICING_SAMPLE_MIN = 4       # s8_c16_t4_inflation_sd3.yaml:22
BYTEDANCE_VAE_TEMPORAL_DOWNSAMPLE = 4  # infer.py:230
BYTEDANCE_VAE_SPATIAL_DOWNSAMPLE = 8   # infer.py:231
BYTEDANCE_SCHEDULE_T = 1000.0          # configs_3b/main.yaml:65
BYTEDANCE_SPATIAL_DIVISOR = 16         # inference_seedvr2_3b.py:241
BYTEDANCE_720P_REF_AREA = 45 * 80      # dit_v2/window.py:32
BYTEDANCE_MAX_TEMPORAL_WINDOW = 30     # dit_v2/window.py:35
BYTEDANCE_ROPE_MAX_FREQ = 256          # dit_v2/rope.py:31
BYTEDANCE_SINUSOIDAL_DIM = 256         # dit_3b/nadit.py:120
BYTEDANCE_IMG_SHIFT_FIT = (256 * 256, 1.0, 1024 * 1024, 3.2)            # infer.py:242
BYTEDANCE_VID_SHIFT_FIT = (256 * 256 * 37, 1.0, 1280 * 720 * 145, 5.0)  # infer.py:243

ROPE_THETA = 10000                     # RoFormer, arXiv:2104.09864
CIELAB_DELTA = 6.0 / 29.0              # CIE 15
CIELAB_KAPPA = (29.0 / 3.0) ** 3       # CIE 15
D65_WHITE_X = 0.95047                  # CIE D65 Xn
D65_WHITE_Z = 1.08883                  # CIE D65 Zn
WAVELET_DECOMP_LEVELS = 5              # StableSR / GIMP-Krita wavelet
SRGB_TO_XYZ_D65 = (                    # IEC 61966-2-1
    (0.4124564, 0.3575761, 0.1804375),
    (0.2126729, 0.7151522, 0.0721750),
    (0.0193339, 0.1191920, 0.9503041),
)
XYZ_TO_SRGB_D65 = (                    # IEC 61966-2-1
    (3.2404542, -1.5371385, -0.4985314),
    (-0.9692660, 1.8760108, 0.0415560),
    (0.0556434, -0.2040259, 1.0572252),
)
