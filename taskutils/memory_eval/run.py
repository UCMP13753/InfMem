# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import time
from dataclasses import dataclass
import sys

sys.stdout.reconfigure(line_buffering=True)
DASH_PORT = os.getenv("DASH_PORT", "8265")
SERVE_PORT = os.getenv("SERVE_PORT", "8000")
MODELROOT = os.getenv("MODELROOT", "/mnt/hdfs/models")


@dataclass
class ENV:
    # config for direct generation
    MAX_INPUT_LEN: int = 120000
    MAX_OUTPUT_LEN: int = 10000
    # Config for memory agent
    RECURRENT_MAX_CONTEXT_LEN: int = None
    RECURRENT_CHUNK_SIZE: int = None
    RECURRENT_MAX_NEW: int = None
    ENABLE_THINK: bool = False
    EARLY_STOP: int = None
    def setenv(self):
        if not hasattr(self, "_environ"):
            self._environ = {}
        for k, v in self.__dict__.items():
            if v is not None and k != "_environ":
                os.environ[k] = str(v)
                self._environ[k] = str(v)
                print(f"set {k}={v}")

    def unsetenv(self):
        for k in self._environ:
            os.environ[k] = self._environ[k]
        self._environ = {}


# for ruler hqa, we just control the number of distractive wiki items instead the context length
# 50~7K tokens, 100~14K tokens and so on.

RULER_HQA_TESTS = [200, 400, 800, 1600, 3200, 6400] # 200, 400, 800, 1600, 3200, 6400

# for other ruler task, we use the standard synthetic scripts for convenient and control the context length.
RULER_TASKS = [
    "qa_1",
    "qa_3",
    "qa_4",
]
RULER_PROMPT_LENGTH = [8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
RULER_GENERRAL_TESTS = [(task, length) for task in RULER_TASKS for length in RULER_PROMPT_LENGTH]
LONGBENCH_TESTS = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique",
                    "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht",
                    "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
import subprocess


class Config:
    SERVE_TAG = "__serve"

    def __init__(self, name, ckpt, model, method, env, concur=1024):
        self.name = name
        self.ckpt = ckpt
        from pathlib import Path

        self.model = model
        self.method = method
        # self.tp = tp
        self.env = env
        self.concur = concur
        self.test_process = {}

    def serve(self, wait=True):
        serve_script = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..", "serve/llm070.py"))
        cmd = f"python {serve_script} --model {self.ckpt} --tp {self.tp}"
        print("serving command:")
        print(cmd)
        if wait:
            os.system(f"yes | serve shutdown -a http://localhost:{DASH_PORT}")
            # setsid so that it can be interrupted
            serve_p = subprocess.Popen(cmd.split(), preexec_fn=os.setsid)
            self.test_process[self.SERVE_TAG] = serve_p
            while True:
                print("try to conntect...")
                p = subprocess.run(["curl", "-m", "100000000", f"http://127.0.0.1:{SERVE_PORT}/v1/models"], capture_output=True)
                if p.returncode != 0:
                    print("waiting...")
                    time.sleep(5)
                elif rf'"id":"{self.model}"' not in p.stdout.decode():
                    print("model not found, maybe shutting down previous server...")
                    time.sleep(5)
                else:
                    print("connected")
                    break
        else:
            p = subprocess.run(["curl", "-m", "10", f"http://127.0.0.1:{SERVE_PORT}/v1/models"], capture_output=True)
            if p.returncode != 0:
                print("server not started")
                exit(1)
        print(p.stdout)

    def run(self, tests, serve=True, force=False):
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        self.env.setenv()
        concur = self.concur
        for test in tests:
            if test in RULER_HQA_TESTS:
                cmd = f"""python ruler_hqa.py --model {self.model}\
                    --length {test} \
                    --save_dir results/ruler_hqa_{test} \
                    --save_file {self.name} \
                    --tokenizer {self.ckpt} \
                    --api {self.method} \
                    --n_proc {concur}"""
            elif test in RULER_GENERRAL_TESTS:
                cmd = f"""python ruler_general.py --model {self.model}\
                    --split {test[0]} \
                    --length {test[1]} \
                    --save_dir results/ruler_{test[0]}_{test[1]} \
                    --save_file {self.name} \
                    --tokenizer {self.ckpt} \
                    --api {self.method} \
                    --n_proc {concur}"""

            elif test in LONGBENCH_TESTS:
                cmd = f"""python longbench_task.py --model {self.model}\
                    --split {test}  \
                    --save_dir results/longbench_{test} \
                    --save_file {self.name} \
                    --tokenizer {self.ckpt} \
                    --api {self.method} \
                    --n_proc {concur}"""
                
            else:
                print("=" * 20 + f"Not Implemented Task {test}, please check" + "=" * 20)
                continue
            if force:
                cmd += " --force"
            p = subprocess.Popen(cmd, shell=True)
            self.test_process[test] = p
            p.wait()
            self.test_process[test].wait()
        self.env.unsetenv()
        if serve:
            os.killpg(os.getpgid(self.test_process[self.SERVE_TAG].pid), 2)
            try:
                self.test_process[self.SERVE_TAG].wait(30)
            except:
                self.test_process[self.SERVE_TAG].kill()
        print("all tests finished")

    def __del__(self):
        for k, p in self.test_process.items():
            if k == self.SERVE_TAG:
                os.killpg(os.getpgid(p.pid), 2)
            else:
                p.kill()


infmem = Config(
    name="infmem_4B_framework",
    ckpt=f"Qwen/Qwen3-4B",
    model="qwen",
    method="infmem",
    concur=256,
    env=ENV(RECURRENT_MAX_CONTEXT_LEN=100000000000, RECURRENT_CHUNK_SIZE=5000, RECURRENT_MAX_NEW=1024, ENABLE_THINK=True,EARLY_STOP=3),
)


memagent = Config(
    name="memagent_4B_framework",
    ckpt=f"Qwen/Qwen3-4B",
    model="qwen",
    method="recurrent",
    concur=256,
    env=ENV(RECURRENT_MAX_CONTEXT_LEN=100000000000, RECURRENT_CHUNK_SIZE=5000, RECURRENT_MAX_NEW=1024,ENABLE_THINK=False),
)

test_openai=Config(
    name="Qwen3_Next",
    ckpt="Qwen/Qwen3-4B",
    model="qwen",
    method="openai",
    concur=256,
    env=ENV(),
)

test_rag=Config(
    name="Qwen-4B-rag",
    ckpt="Qwen/Qwen3-4B",
    model="qwen",
    method="rag",
    concur=256,
    env=ENV()
)
CONFIGS = [
    # OURS
    # test_rag,
    # memagent,
    infmem,
    # test_openai,
]

def run_ruler_hqa():
    for c in CONFIGS:
        task = RULER_HQA_TESTS
        c.run(task, serve=False, force=False)


def run_ood_tasks():
    for c in CONFIGS:
        subset = [
            "qa_1",
            "qa_3",
            "qa_4",
        ]
        
        lengths = [32768, 65536, 131072, 262144, 524288, 1048576] # 32768, 65536, 131072
        task = [(s, l) for s in subset for l in lengths]
        c.run(task, serve=False, force=False)

def run_longbench_tasks():
    for c in CONFIGS:
        subset = [
            "narrativeqa",
            "hotpotqa",
            "2wikimqa",
            "qasper",
            "musique", 
        ]
        
        task = [s for s in subset]
        c.run(task, serve=False, force=False)
if __name__ == "__main__":
    print(f"{SERVE_PORT=}, {DASH_PORT=}, {MODELROOT=}")
    run_ruler_hqa()
    run_ood_tasks()
    run_longbench_tasks()

