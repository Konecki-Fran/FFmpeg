/**
 * @file
 * vf_rgbstats.c
 *
 * Filter:
 *  1) Count pixels matching target color (per frame)
 *  2) Rolling sum of (1) across last 25 frames
 *  3) Cumulative paint: once a pixel matches target color, it stays painted forever
 *  4) Save each processed frame as JPEG (picture-N.jpg)
 *
 * Usage:
 *   ./ffmpeg -i input.mp4 -vf "format=rgb24,rgbstats=color=black" -f null -
 *
 * Notes:
 *  - Filter accepts only RGB24 input. (enforced by query_formats)
 *  - MJPEG encoder in FFmpeg 4.4.x supports pixel formats: yuvj420p, yuvj422p, yuvj444p. (JPEGs are stored as YCbCr)
 */

#include <stdio.h>
#include <errno.h>
#include <inttypes.h>

#include "libavutil/colorspace.h"
#include "libavutil/opt.h"
#include "libavutil/mem.h"
#include "libavutil/pixdesc.h"
#include "libavutil/error.h"

#include "libavfilter/avfilter.h"
#include "libavfilter/internal.h"
#include "libavfilter/video.h"
#include "libavfilter/formats.h"

#include "libavcodec/avcodec.h"

#include "libswscale/swscale.h"

#define RGBSTATS_FILENAME_MAX 64
#define RGBSTATS_ROLLING_N 25

typedef struct RgbStatsContext
{
    const AVClass *class;
    uint8_t color[4];
    int width;                // pixels in frame row
    int height;               // pixels in frame column
    uint8_t *cumulative_mask; // butckets of 8 bits, 1 bit per pixel
    size_t mask_bytes;        // number of buckets for storing bits/pixels
    size_t pixel_count;       // width * height
    int64_t frame_index;
    int matches_per_frame[RGBSTATS_ROLLING_N]; // stores matches of last 25 frames
    int64_t matches_sum;                       // total matches sum of last 25 frames
    AVCodecContext *jpeg_ctx;                  // encoder state
    struct SwsContext *sws;                    // pixel-format converter
    AVFrame *enc_frame;                        // dest frame buffer (yuvj444p)
} RgbStatsContext;

#define OFFSET(x) offsetof(RgbStatsContext, x)
#define FLAGS AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_FILTERING_PARAM

static const AVOption rgbstats_options[] = {
    {"color", "target color", OFFSET(color), AV_OPT_TYPE_COLOR, {.str = "black"}, 0, 0, FLAGS},
    {NULL}};

AVFILTER_DEFINE_CLASS(rgbstats);

static inline void set_mask_bit(uint8_t *cumulative_mask, size_t pixel)
{
    cumulative_mask[pixel >> 3] |= (uint8_t)(1u << (pixel & 7));
}

static inline int get_mask_bit(const uint8_t *cumulative_mask, size_t pixel)
{
    return (cumulative_mask[pixel >> 3] >> (pixel & 7)) & 1u;
}

// Enforce RGB24 input
static int query_formats(AVFilterContext *ctx)
{
    static const enum AVPixelFormat pix_fmts[] = {AV_PIX_FMT_RGB24, AV_PIX_FMT_NONE};
    return ff_set_common_formats(ctx, ff_make_format_list(pix_fmts));
}

// MJPEG encoder init: FFmpeg 4.4.x mjpeg supports yuvj420p/yuvj422p/yuvj444p
static int init_jpeg_encoder(RgbStatsContext *state, AVRational rational)
{
    const AVCodec *codec;
    int result;

    codec = avcodec_find_encoder(AV_CODEC_ID_MJPEG);
    if (!codec)
        return AVERROR_ENCODER_NOT_FOUND;

    state->jpeg_ctx = avcodec_alloc_context3(codec);
    if (!state->jpeg_ctx)
        return AVERROR(ENOMEM);

    state->jpeg_ctx->pix_fmt = AV_PIX_FMT_YUVJ444P;
    state->jpeg_ctx->width = state->width;
    state->jpeg_ctx->height = state->height;
    state->jpeg_ctx->color_range = AVCOL_RANGE_JPEG;

    if (rational.num && rational.den)
        state->jpeg_ctx->time_base = av_inv_q(rational);
    else
        state->jpeg_ctx->time_base = (AVRational){1, 25};

    result = avcodec_open2(state->jpeg_ctx, codec, NULL);
    if (result < 0)
        return result;

    return 0;
}

static int save_frame_as_jpeg(RgbStatsContext *state, AVFrame *frame)
{
    char filename[RGBSTATS_FILENAME_MAX];
    FILE *f;
    int result;
    AVPacket *pkt;

    snprintf(filename, sizeof(filename), "picture-%" PRId64 ".jpg", state->frame_index);

    result = avcodec_send_frame(state->jpeg_ctx, frame);
    if (result < 0)
        return result;

    pkt = av_packet_alloc();
    if (!pkt)
        return AVERROR(ENOMEM);

    for (;;)
    {
        result = avcodec_receive_packet(state->jpeg_ctx, pkt);
        if (result == AVERROR(EAGAIN) || result == AVERROR_EOF)
        {
            av_packet_free(&pkt);
            return 0;
        }
        if (result < 0)
        {
            av_packet_free(&pkt);
            return result;
        }

        f = fopen(filename, "wb");
        if (!f)
        {
            av_packet_free(&pkt);
            return AVERROR(errno);
        }

        if (fwrite(pkt->data, 1, pkt->size, f) != (size_t)pkt->size)
        {
            fclose(f);
            av_packet_free(&pkt);
            return AVERROR(EIO);
        }

        fclose(f);
        av_packet_unref(pkt);

        // MJPEG: one packet per frame is expected
        av_packet_free(&pkt);
        return 0;
    }
}

static av_cold int init(AVFilterContext *ctx)
{
    RgbStatsContext *state;
    int i;

    state = ctx->priv;
    state->frame_index = 0;
    state->matches_sum = 0;
    for (i = 0; i < RGBSTATS_ROLLING_N; i++)
        state->matches_per_frame[i] = 0;

    state->jpeg_ctx = NULL;
    state->sws = NULL;
    state->enc_frame = NULL;
    state->cumulative_mask = NULL;

    return 0;
}

static av_cold void uninit(AVFilterContext *ctx)
{
    RgbStatsContext *state = ctx->priv;

    av_freep(&state->cumulative_mask);

    if (state->jpeg_ctx)
        avcodec_free_context(&state->jpeg_ctx);

    if (state->sws)
        sws_freeContext(state->sws);
    state->sws = NULL;

    av_frame_free(&state->enc_frame);
}

static int config_input(AVFilterLink *inlink)
{
    AVFilterContext *ctx;
    RgbStatsContext *state;
    int result;
    const int *coeffs;

    ctx = inlink->dst;
    state = ctx->priv;

    state->width = inlink->w;
    state->height = inlink->h;
    state->pixel_count = (size_t)state->width * (size_t)state->height;
    state->mask_bytes = (state->pixel_count + 7) / 8;

    state->cumulative_mask = av_mallocz(state->mask_bytes);
    if (!state->cumulative_mask)
        return AVERROR(ENOMEM);

    result = init_jpeg_encoder(state, inlink->frame_rate);
    if (result < 0)
        return result;

    state->enc_frame = av_frame_alloc();
    if (!state->enc_frame)
        return AVERROR(ENOMEM);

    state->enc_frame->format = AV_PIX_FMT_YUV444P;
    state->enc_frame->width = state->width;
    state->enc_frame->height = state->height;

    // Explicit full-range on the frame (matches yuvj* intent)
    state->enc_frame->color_range = AVCOL_RANGE_JPEG;

    result = av_frame_get_buffer(state->enc_frame, 32);
    if (result < 0)
        return result;

    // Convert RGB24 -> yuv444p
    state->sws = sws_getContext(state->width, state->height, AV_PIX_FMT_RGB24,
                                state->width, state->height, AV_PIX_FMT_YUV444P,
                                SWS_BILINEAR, NULL, NULL, NULL);
    if (!state->sws)
        return AVERROR(ENOMEM);

    /*
     * Ensure RGB->YUV conversion uses full-range and standard coefficients.
     * This avoids "looks like YUV colors" issues that typically come from
     * incorrect range/coefficient assumptions.
     *
     * srcRange=1 because RGB is full-range.
     * dstRange=1 because yuvj* is full-range JPEG.
     */
    coeffs = sws_getCoefficients(SWS_CS_DEFAULT);
    result = sws_setColorspaceDetails(state->sws,
                                      coeffs, 1,
                                      coeffs, 1,
                                      0, 1 << 16, 1 << 16);
    if (result < 0)
        return result;

    return 0;
}

static int filter_frame(AVFilterLink *inlink, AVFrame *frame)
{
    AVFilterContext *ctx;
    RgbStatsContext *state;
    uint8_t target_color[3];
    uint8_t *data;
    int linesize;
    int matches;
    unsigned idx;
    int result;
    int y, x;

    ctx = inlink->dst;
    state = ctx->priv;

    if (frame->width != state->width || frame->height != state->height)
    {
        av_log(ctx, AV_LOG_ERROR, "Frame dimension change not supported (%dx%d -> %dx%d)\n",
               state->width, state->height, frame->width, frame->height);
        return AVERROR(EINVAL);
    }

    target_color[0] = state->color[0];
    target_color[1] = state->color[1];
    target_color[2] = state->color[2];

    data = frame->data[0];
    linesize = frame->linesize[0];

    if (linesize < 0)
    {
        data += (ptrdiff_t)(frame->height - 1) * linesize;
        linesize = -linesize;
    }

    matches = 0;

    // Count matches and update cumulative mask
    for (y = 0; y < frame->height; y++)
    {
        uint8_t *row = data + (ptrdiff_t)y * linesize;
        for (x = 0; x < frame->width; x++)
        {
            uint8_t *px_rgb = row + (ptrdiff_t)x * 3;
            size_t pixel = (size_t)y * (size_t)frame->width + (size_t)x;

            if (px_rgb[0] == target_color[0] &&
                px_rgb[1] == target_color[1] &&
                px_rgb[2] == target_color[2])
            {
                matches++;
                set_mask_bit(state->cumulative_mask, pixel);
            }
        }
    }

    idx = (unsigned)(state->frame_index % RGBSTATS_ROLLING_N);
    state->matches_sum -= state->matches_per_frame[idx];
    state->matches_per_frame[idx] = matches;
    state->matches_sum += matches;

    // Apply cumulative paint mask
    for (y = 0; y < frame->height; y++)
    {
        uint8_t *row = data + (ptrdiff_t)y * linesize;
        for (x = 0; x < frame->width; x++)
        {
            size_t pixel = (size_t)y * (size_t)frame->width + (size_t)x;
            if (get_mask_bit(state->cumulative_mask, pixel))
            {
                uint8_t *px_rgb = row + (ptrdiff_t)x * 3;
                px_rgb[0] = target_color[0];
                px_rgb[1] = target_color[1];
                px_rgb[2] = target_color[2];
            }
        }
    }

    fprintf(stdout, "frame:%" PRId64 " pixels_match:%d last_25_total:%" PRId64 "\n",
            state->frame_index, matches, state->matches_sum);
    fflush(stdout);

    // Propagate color metadata where available
    state->enc_frame->color_range = AVCOL_RANGE_JPEG;
    state->enc_frame->colorspace = (frame->colorspace != AVCOL_SPC_UNSPECIFIED) ? frame->colorspace : AVCOL_SPC_BT709;
    state->enc_frame->color_primaries = (frame->color_primaries != AVCOL_PRI_UNSPECIFIED) ? frame->color_primaries : AVCOL_PRI_BT709;
    state->enc_frame->color_trc = (frame->color_trc != AVCOL_TRC_UNSPECIFIED) ? frame->color_trc : AVCOL_TRC_BT709;

    // Convert RGB24 -> yuvj444p and encode
    result = av_frame_make_writable(state->enc_frame);
    if (result < 0)
        return result;

    sws_scale(state->sws,
              (const uint8_t *const *)frame->data, frame->linesize,
              0, frame->height,
              state->enc_frame->data, state->enc_frame->linesize);

    result = save_frame_as_jpeg(state, state->enc_frame);
    if (result < 0)
    {
        av_log(ctx, AV_LOG_ERROR, "Failed to write JPEG for frame %" PRId64 ": %s\n",
               state->frame_index, av_err2str(result));
        return result;
    }

    state->frame_index++;

    // Pass frame downstream
    return ff_filter_frame(inlink->dst->outputs[0], frame);
}

static const AVFilterPad rgbstats_inputs[] = {
    {
        .name = "default",
        .type = AVMEDIA_TYPE_VIDEO,
        .filter_frame = filter_frame,
        .config_props = config_input,
    },
    {NULL}};

static const AVFilterPad rgbstats_outputs[] = {
    {
        .name = "default",
        .type = AVMEDIA_TYPE_VIDEO,
    },
    {NULL}};

AVFilter ff_vf_rgbstats = {
    .name = "rgbstats",
    .description = NULL_IF_CONFIG_SMALL(
        "Count target-color pixels, update and apply cumulative paint, rolling sum for matches in last 25 frames, and save modified frames as JPEG."),
    .priv_size = sizeof(RgbStatsContext),
    .priv_class = &rgbstats_class,
    .init = init,
    .uninit = uninit,
    .query_formats = query_formats,
    .inputs = rgbstats_inputs,
    .outputs = rgbstats_outputs,
};
