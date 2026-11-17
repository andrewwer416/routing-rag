"""Require working CUDA before starting the pinned Kev 1.0 server.

Kev 1.0 predates upstream's --device flag. This wrapper checks allocation and matrix
multiplication before its auto-device selection, so a missing GPU fails startup.
"""

import torch


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Kev-4B requires CUDA; check the GPU device and driver mounts")
    probe = torch.ones((2, 2), device="cuda", dtype=torch.bfloat16)
    result = probe @ probe
    torch.cuda.synchronize()
    if result[0, 0].item() != 2:
        raise RuntimeError("CUDA matrix multiplication failed")
    del probe, result
    torch.cuda.empty_cache()
    print(f"CUDA verified: {torch.cuda.get_device_name(0)} (torch {torch.__version__})", flush=True)
    from kev.serve import main as serve

    serve()


if __name__ == "__main__":
    main()
