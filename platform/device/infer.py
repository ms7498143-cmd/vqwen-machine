#!/usr/bin/env python3
"""
Inference job for the Python device: runs the registered model locally and
reports real tokens/sec. Generation is a genuine autoregressive loop over ONNX
Runtime with a key/value cache. Nothing here is simulated.
"""
import argparse, json, os, time
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")
MODEL_REPO = "onnx-community/Qwen2.5-0.5B-Instruct"


class Model:
    def __init__(self, model_dir=MODEL_DIR):
        t0 = time.perf_counter()
        self.tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(
            os.path.join(model_dir, "onnx", "model_q4f16.onnx"),
            sess_options=opts, providers=["CPUExecutionProvider"])
        self.in_names = [i.name for i in self.sess.get_inputs()]
        self.out_names = [o.name for o in self.sess.get_outputs()]
        self.past_in = [n for n in self.in_names if n.startswith("past_key_values")]
        self.present_out = [n for n in self.out_names if n.startswith("present")]
        self.load_s = time.perf_counter() - t0

    def _empty_past(self):
        return {n: np.zeros((1, 2, 0, 64), dtype=np.float32) for n in self.past_in}

    def chat(self, prompt, max_new=48):
        text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        ids = self.tok.encode(text).ids
        input_ids = np.array([ids], dtype=np.int64)
        attention = np.ones((1, len(ids)), dtype=np.int64)
        position = np.arange(len(ids), dtype=np.int64).reshape(1, -1)
        past = self._empty_past()
        past_len = 0
        out_tokens = []

        t0 = time.perf_counter()
        cur_ids, cur_att, cur_pos = input_ids, attention, position
        for _ in range(max_new):
            feeds = {"input_ids": cur_ids, "attention_mask": cur_att, "position_ids": cur_pos}
            feeds.update(past)
            outs = self.sess.run(None, feeds)
            by_name = dict(zip(self.out_names, outs))
            next_id = int(np.argmax(by_name["logits"][0, -1, :]))
            if next_id == self.tok.token_to_id("<|im_end|>"):
                break
            out_tokens.append(next_id)
            for pi, po in zip(self.past_in, self.present_out):
                past[pi] = by_name[po]
            past_len += cur_ids.shape[1]
            cur_ids = np.array([[next_id]], dtype=np.int64)
            cur_att = np.ones((1, past_len + 1), dtype=np.int64)
            cur_pos = np.array([[past_len]], dtype=np.int64)

        gen_s = time.perf_counter() - t0
        return {
            "job": "infer",
            "model": MODEL_REPO,
            "quantization": "q4f16",
            "load_s": round(self.load_s, 2),
            "gen_s": round(gen_s, 3),
            "tokens": len(out_tokens),
            "tokens_per_sec": round(len(out_tokens) / gen_s, 2) if gen_s else 0,
            "reply": self.tok.decode(out_tokens, skip_special_tokens=True).strip(),
        }

    def stream(self, prompt, max_new=48):
        """Yield tokens as they are produced.

        Each item is ("token", text) for a decoded piece, then a final
        ("done", stats) where stats carries the measured generation rate.
        Decoding is incremental: only the new suffix is emitted, so the client
        can append text without re-rendering.
        """
        text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        ids = self.tok.encode(text).ids
        cur_ids = np.array([ids], dtype=np.int64)
        cur_att = np.ones((1, len(ids)), dtype=np.int64)
        cur_pos = np.arange(len(ids), dtype=np.int64).reshape(1, -1)
        past = self._empty_past()
        past_len = 0
        out_tokens = []
        emitted = ""

        t0 = time.perf_counter()
        for _ in range(max_new):
            feeds = {"input_ids": cur_ids, "attention_mask": cur_att, "position_ids": cur_pos}
            feeds.update(past)
            outs = self.sess.run(None, feeds)
            by_name = dict(zip(self.out_names, outs))
            next_id = int(np.argmax(by_name["logits"][0, -1, :]))
            if next_id == self.tok.token_to_id("<|im_end|>"):
                break
            out_tokens.append(next_id)
            full = self.tok.decode(out_tokens, skip_special_tokens=True)
            if full.startswith(emitted):
                piece = full[len(emitted):]
            else:
                piece = full
            emitted = full
            if piece:
                yield ("token", piece)
            for pi, po in zip(self.past_in, self.present_out):
                past[pi] = by_name[po]
            past_len += cur_ids.shape[1]
            cur_ids = np.array([[next_id]], dtype=np.int64)
            cur_att = np.ones((1, past_len + 1), dtype=np.int64)
            cur_pos = np.array([[past_len]], dtype=np.int64)

        gen_s = time.perf_counter() - t0
        yield ("done", {
            "job": "infer-stream",
            "model": MODEL_REPO,
            "quantization": "q4f16",
            "load_s": round(self.load_s, 2),
            "gen_s": round(gen_s, 3),
            "tokens": len(out_tokens),
            "tokens_per_sec": round(len(out_tokens) / gen_s, 2) if gen_s else 0,
        })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="What does a CPU do? One sentence.")
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    m = Model()
    r = m.chat(a.prompt, a.max_new)
    if a.json:
        print(json.dumps(r, indent=2))
    else:
        print(f"tok/s: {r['tokens_per_sec']}  ({r['tokens']} tokens in {r['gen_s']}s)\nreply: {r['reply']}")


if __name__ == "__main__":
    main()
