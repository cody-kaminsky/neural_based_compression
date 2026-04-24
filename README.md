# Neural-Based Compression

Neural autoencoder video compression targeting FPGA deployment on drone platforms. The system replaces traditional H.265 with a learned end-to-end codec optimised for the low-bitrate, low-latency constraints of drone video downlinks.

## Project Status

**Phase 0 — In Progress:** establishing training pipeline, reference codec, and decoder infrastructure.

## Architecture

The codec is a learned image compression autoencoder:

- **Encoder** (runs on drone FPGA): convolutional analysis transform → quantisation → entropy coding (rANS, 64-stream interleaved)
- **Decoder** (runs ground-side in Python/FPGA): entropy decoding → synthesis transform → reconstructed frames
- **Training**: end-to-end rate-distortion optimisation with MS-SSIM + bitrate loss

## Repository Layout

```
decoder/        Python entropy decoder and synthesis transform
train/          Training scripts, dataset loaders, loss functions
  modules/      Neural network modules (encoder, decoder, hyperprior)
eval/           BD-rate and perceptual quality evaluation
export/         ONNX / VHDL export utilities
tests/          Unit and integration tests
docs/           Architecture reference and implementation plans
reference_vectors/  Golden I/O vectors for bitstream compliance testing
```

## Quick Start

```bash
pip install -r requirements.txt
pytest tests/
```

## Key Parameters

| Parameter | Value |
|-----------|-------|
| ANS state range | [2²³, 256·2²³) |
| Probability precision M | 2¹² = 4096 |
| ANS streams | 64 (round-robin interleaved) |
| Target bitrate | 0.1 – 2.0 bpp |

## References

- Architecture details: `docs/architecture_reference.txt`
- Implementation plan: `docs/implementation_plan.txt`
- Phase 0 task list: `docs/phase0_task_list.txt`
