import csv
import sys
import threading
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path

import ollama
import psutil
from PIL import Image

DEFAULT_MODEL = "qwen2.5vl:3b"
MAX_SIDE = 640  # images are shrunk to save time and memory
NUM_CTX = 2048
LOG = Path("eval/vlm_benchmark.csv")


def ollama_ram_gb() -> float:
    total = 0
    for p in psutil.process_iter(["name", "memory_info"]):
        try:
            if "ollama" in (p.info["name"] or "").lower():
                total += p.info["memory_info"].rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total / 1e9


class RamSampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.peak = 0.0
        self._halt = threading.Event()

    def run(self):
        while not self._halt.is_set():
            self.peak = max(self.peak, ollama_ram_gb())
            time.sleep(0.5)

    def stop(self):
        self._halt.set()


def prepare(path: str) -> bytes:
    img = Image.open(path).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def run_test(model: str, name: str, question: str, image_path: str | None = None) -> str:
    message = {"role": "user", "content": question}
    if image_path:
        message["images"] = [prepare(image_path)]

    sampler = RamSampler()
    sampler.start()
    t0 = time.time()
    resp = ollama.chat(
        model=model,
        messages=[message],
        options={"num_ctx": NUM_CTX, "temperature": 0.2, "num_predict": 300},
    )
    wall = time.time() - t0
    sampler.stop()

    answer = resp["message"]["content"]
    gen_tokens = resp.get("eval_count") or 0
    gen_s = (resp.get("eval_duration") or 0) / 1e9
    load_s = (resp.get("load_duration") or 0) / 1e9
    prompt_tokens = resp.get("prompt_eval_count") or 0
    tps = gen_tokens / gen_s if gen_s else 0

    print(f"\n=== {name} | {model}")
    print(f"wall={wall:.1f}s load={load_s:.1f}s prompt_tokens={prompt_tokens} "
          f"gen_tokens={gen_tokens} speed={tps:.1f} tok/s peak_ollama_RAM={sampler.peak:.1f} GB")
    print(answer)

    LOG.parent.mkdir(exist_ok=True)
    new_file = not LOG.exists()
    with open(LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["timestamp", "model", "test", "wall_s", "load_s", "prompt_tokens",
                        "gen_tokens", "tok_per_s", "peak_ollama_ram_gb", "max_side", "num_ctx"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), model, name, f"{wall:.1f}",
                    f"{load_s:.1f}", prompt_tokens, gen_tokens, f"{tps:.1f}",
                    f"{sampler.peak:.2f}", MAX_SIDE, NUM_CTX])
    return answer


if __name__ == "__main__":
    img = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL

    run_test(model, "text_only", "Explique en deux phrases à quoi sert un voyant d'alarme sur une machine.")
    run_test(model, "describe", "Décris précisément ce que tu vois sur cette photo.", img)
    run_test(model, "leds",
             "Quels voyants ou indicateurs sont visibles ? Donne leur couleur et leur position. "
             "Si tu n'es pas sûr, dis-le clairement.", img)