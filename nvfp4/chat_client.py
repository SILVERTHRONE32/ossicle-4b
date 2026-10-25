"""Talk about an audio file with two vLLM servers: Ossicle listens, the thinking model talks.
    vllm serve . --served-model-name ossicle --port 8001 --limit-mm-per-prompt '{"audio": 2}' --max-num-batched-tokens 8192
    vllm serve thinker --served-model-name thinker --port 8000
    python chat_client.py clip.wav"""
import base64, io, json, sys, urllib.request

import soundfile as sf

OSSICLE, THINKER = "http://127.0.0.1:8001", "http://127.0.0.1:8000"
FACETS = [("Genre", "What genre is this?"), ("Instruments", "What instruments are playing?"),
          ("Voice", "Is there singing or speech? If so, what is it like?"), ("Mood", "What is the mood?")]
INTRO = ("You are talking with a user about an audio recording you have listened to. Below is what you heard, in your own words (it can contain mistakes: you may have missed quiet sounds "
         "or mixed up instruments). Answer as someone who heard the recording; never mention notes, an audio model, or where your information comes from. "
         "Do not invent details you do not actually have, such as timestamps, instruments or events; if you are not sure, or you did not catch something, say so plainly. "
         "Do not just agree with a leading question: if what you heard does not support it, say that. Be conversational and answer in as much detail as the question deserves. "
         "For general questions about music, sound or anything else, use your own knowledge.\n\nWHAT YOU HEARD:\n")


def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))


def wav_b64(samples, sr):
    buf = io.BytesIO(); sf.write(buf, samples, sr, format="WAV"); return base64.b64encode(buf.getvalue()).decode()


def ask_ossicle(b64, question, max_tokens=120):
    body = {"model": "ossicle", "temperature": 0, "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": [{"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64," + b64}}, {"type": "text", "text": question}]}]}
    return post(OSSICLE + "/v1/chat/completions", body)["choices"][0]["message"]["content"].strip()


def listen(path):
    """Per-minute notes (sounds and speech) plus short answers about the whole clip."""
    w, sr = sf.read(path, dtype="float32", always_2d=True); w = w.mean(1)[: 240 * sr]
    lines = [f"Audio ({path}, {len(w) / sr:.0f} s analysed):"]
    for s0 in range(0, len(w), 60 * sr):
        seg = w[s0: s0 + 60 * sr]
        if len(seg) < sr: continue
        b = wav_b64(seg, sr)
        lines.append(f"  [{s0 // sr // 60}:{s0 // sr % 60:02d}] sounds: {ask_ossicle(b, 'Describe the sounds or music in this audio in two sentences.', 90)} | "
                     f"speech: {ask_ossicle(b, 'Transcribe any speech in this audio exactly. If there is none, say there is no speech.', 120)}")
    whole = wav_b64(w[: 120 * sr], sr)
    lines += [f"  {label}: {ask_ossicle(whole, q, 40)}" for label, q in FACETS]
    return "\n".join(lines)


def prompt(profile, history, message):
    out = ""
    for k, m in enumerate(history + [{"role": "user", "content": message}]):
        text = (INTRO + profile + "\n\nUSER: " if k == 0 else "") + m["content"]
        out += f"<|im_start|>{m['role']}\n" + ("<think>\n\n</think>\n\n" if m["role"] == "assistant" else "") + text + "<|im_end|>\n"
    return out + "<|im_start|>assistant\n<think>\n"


def think(p, thinking_tokens=600, answer_tokens=500):
    def call(text, n):
        return post(THINKER + "/v1/completions", {"model": "thinker", "prompt": text, "max_tokens": n, "temperature": 0.6, "top_p": 0.95})["choices"][0]["text"]
    thought = call(p, thinking_tokens)
    return (thought.split("</think>", 1)[1] if "</think>" in thought else call(p + thought + "\n</think>\n\n", answer_tokens)).strip()


if __name__ == "__main__":
    profile, history = listen(sys.argv[1]), []
    print("(listened; ask away, empty line to quit)")
    while (msg := input("you> ").strip()):
        reply = think(prompt(profile, history, msg))
        history += [{"role": "user", "content": msg}, {"role": "assistant", "content": reply}]
        print("ossicle>", reply)
