"""Start local Qwen on Podman, including the Ubuntu 22.04 CNI workaround."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main(default_router="qwen"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", action="store_true", help="Explicitly disable WSL GPU passthrough")
    parser.add_argument("--router", choices=["qwen", "kev"], default=default_router)
    args = parser.parse_args()
    if args.cpu and args.router == "kev":
        parser.error("This Kev-4B deployment requires CUDA; use --router qwen --cpu for CPU mode")
    root = Path(__file__).resolve().parents[1]
    compose = root / ".venv/bin/podman-compose"
    if not compose.exists():
        sys.exit("Install the containers extra first: .venv/bin/pip install -e '.[containers]'")
    network = "agentic-rag-openwebui-demo_default"
    if subprocess.run(["podman", "network", "exists", network], check=False).returncode:
        subprocess.run(["podman", "network", "create", network], check=True)
    version = subprocess.check_output(["podman", "--version"], text=True)
    config_home = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    config = config_home / "cni/net.d" / f"{network}.conflist"
    if "version 3.4." in version and config.exists():
        data = json.loads(config.read_text())
        if data.get("cniVersion") == "1.0.0":
            backup = config.with_suffix(".conflist.backup")
            if not backup.exists():
                shutil.copy2(config, backup)
            data["cniVersion"] = "0.4.0"
            config.write_text(json.dumps(data, indent=3) + "\n")
            print(f"Corrected CNI version in {config}; backup: {backup}", flush=True)
    command = [
        str(compose),
        "-f",
        "compose.yaml",
        "-f",
        "compose.podman.yaml",
        "-f",
        "compose.qwen.yaml",
    ]
    wsl_gpu = Path("/dev/dxg").exists() and Path("/usr/lib/wsl/lib/libcuda.so.1").exists()
    if wsl_gpu and not args.cpu:
        command += ["-f", "compose.qwen.wsl-gpu.yaml"]
        print("Using WSL NVIDIA GPU passthrough for Ollama.", flush=True)
    else:
        print("Using CPU inference (WSL GPU unavailable or --cpu selected).", flush=True)
    if args.router == "qwen":
        # A previous Kev service is an orphan in this profile; release its GPU memory.
        subprocess.run(
            [
                "podman",
                "stop",
                "agentic-rag-openwebui-demo_kev_1",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    if args.router == "kev":
        command += ["-f", "compose.kev.yaml"]
        if wsl_gpu:
            command += ["-f", "compose.kev.wsl-gpu.yaml"]
        # Release the previous Ollama planner before loading the larger bf16 Kev backbone.
        old = subprocess.run(
            [
                "podman",
                "exec",
                "agentic-rag-openwebui-demo_retrieval_1",
                "python",
                "-c",
                "import json; from agentic_rag.config import Settings; s=Settings(); "
                "print(json.dumps({'provider':getattr(s,'routing_provider','chat'),"
                "'router':s.routing_model or s.llm_model,'answer':s.llm_model}))",
            ],
            capture_output=True,
            text=True,
        )
        if old.returncode == 0:
            previous = json.loads(old.stdout)
            if previous["provider"] == "chat" and previous["router"] != previous["answer"]:
                subprocess.run(
                    [
                        "podman",
                        "exec",
                        "agentic-rag-openwebui-demo_ollama_1",
                        "ollama",
                        "stop",
                        previous["router"],
                    ],
                    check=True,
                )
    subprocess.run(command + ["up", "-d"], cwd=root, check=True)
    if args.router == "kev":
        print("Waiting for Kev-4B to load on CUDA...", flush=True)
        probe = """
import time, httpx
from agentic_rag.config import Settings
s = Settings()
deadline = time.monotonic() + 900
while time.monotonic() < deadline:
    try:
        r = httpx.get(s.kev_base_url.rstrip('/') + '/v1/models',
                     headers={'Authorization': 'Bearer ' + s.kev_api_key}, timeout=3)
        r.raise_for_status()
        card = r.json()['models'][0]
        if card['device'] != 'cuda':
            raise SystemExit('Kev did not start on CUDA')
        print('Kev ready:', card['run'], card['device'], card['dtype'], flush=True)
        break
    except (httpx.HTTPError, KeyError, IndexError):
        time.sleep(2)
else:
    raise SystemExit('Kev startup timed out; inspect podman logs for the Kev container')
"""
        subprocess.run(
            [
                "podman",
                "exec",
                "-i",
                "agentic-rag-openwebui-demo_retrieval_1",
                "python",
                "-",
            ],
            input=probe,
            text=True,
            check=True,
        )
    # Read the effective configuration, including Compose overrides and .env, so both
    # the planner and answer model are downloaded without duplicating dotenv parsing.
    configured_models = json.loads(
        subprocess.check_output(
            [
                "podman",
                "exec",
                "agentic-rag-openwebui-demo_retrieval_1",
                "python",
                "-c",
                "import json; from agentic_rag.config import Settings; "
                "s=Settings(); print(json.dumps([s.llm_model] + "
                "([s.routing_model or s.llm_model] if s.routing_provider == 'chat' else [])))",
            ],
            text=True,
        )
    )
    for model in dict.fromkeys(configured_models):
        subprocess.run(
            [
                "podman",
                "exec",
                "agentic-rag-openwebui-demo_ollama_1",
                "ollama",
                "pull",
                model,
            ],
            check=True,
        )
        print(f"Warming {model} before the first chat...", flush=True)
        subprocess.run(
            [
                "podman",
                "exec",
                "agentic-rag-openwebui-demo_retrieval_1",
                "python",
                "-c",
                "import httpx,sys; r=httpx.post('http://ollama:11434/api/generate', "
                "json={'model':sys.argv[1],'prompt':'Ready','stream':False,"
                "'options':{'num_predict':1}}, timeout=180); r.raise_for_status()",
                model,
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
