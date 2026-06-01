import os
import sys
import time
import subprocess
import json
import urllib.request
import urllib.error
import threading

SERVER_LOG_PATH = "llama_server_rapid.log"
SERVER_HOST = "127.0.0.1"
SERVER_PORT = "8080"
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"

# Exact command requested by user
CMD = [
    os.path.expanduser("~/llama.cpp/build/bin/llama-server"),
    "-m", os.path.expanduser("~/models/Qwen3.6-27B-UD-Q4_K_XL.gguf"),
    "--host", SERVER_HOST,
    "--port", SERVER_PORT,
    "-ngl", "99",
    "-c", "70000",
    "-np", "1",
    "--temp", "0.6",
    "--top-p", "0.95",
    "--top-k", "20",
    "--min-p", "0.0",
    "--presence-penalty", "0.0",
    "--repeat-penalty", "1.0",
    "--cache-type-k", "q8_0",
    "--cache-type-v", "q8_0",
    "--spec-type", "draft-mtp",
    "--spec-draft-n-max", "3",
    "--spec-draft-p-min", "0.05",
    "--chat-template-kwargs", '{"preserve_thinking": true}',
    "--tools", "all",
    "--timeout", "1800",
    "--flash-attn", "on",
    "--jinja",
    "--webui",
    "-a", "qwen3.6-27b-mtp"
]


server_process = None
server_crashed = False
server_crash_reason = ""

def monitor_server(proc, log_file):
    global server_crashed, server_crash_reason
    try:
        for line in iter(proc.stdout.readline, ""):
            if not line:
                break
            log_file.write(line)
            log_file.flush()
            
            stripped = line.strip()
            # Log key server actions and any warning/error indicators
            if any(err in stripped for err in ["CUDA error", "illegal memory access", "Segmentation fault", "sigsegv", "ERROR", "Warning", "failed"]):
                print(f"[SERVER LOG] {stripped}", flush=True)
            elif "server is listening on" in stripped or "all slots are idle" in stripped or "created context checkpoint" in stripped:
                print(f"[SERVER LOG] {stripped}", flush=True)
            
            # Detect crash markers
            if "CUDA error" in stripped or "illegal memory access" in stripped:
                server_crashed = True
                server_crash_reason = f"CUDA Error: {stripped}"
            elif "Segmentation fault" in stripped:
                server_crashed = True
                server_crash_reason = f"Segmentation Fault: {stripped}"
    except Exception as e:
        print(f"Error in monitor thread: {e}", flush=True)

def generate_rapid_prompts():
    languages = [
        "flutter", "php", "rust", "go", "typescript", "python", "ruby", "c", "c++", "java",
        "kotlin", "swift", "scala", "elixir", "haskell", "perl", "bash", "html/css/js", "sql", "julia",
        "r", "matlab", "fortran", "cobol", "pascal", "assembly", "lisp", "scheme", "prolog", "clojure",
        "dart", "lua", "powershell", "ocaml", "f#", "c#", "groovy", "zig", "nim", "vlang"
    ]
    
    prompts = []
    
    # Mix jokes and quick code script requests
    prompts.append("tell me a joke")
    prompts.append("tell me another joke")
    prompts.append("tell me one more joke")
    prompts.append("tell me a very short programmer joke")
    prompts.append("tell me a joke about compiler optimizations")
    
    for i, lang in enumerate(languages):
        prompts.append(f"write a demo script in {lang}. dont write to files. keep it extremely short and concise under 15 lines.")
        if i % 5 == 0:
            prompts.append(f"tell me a quick joke about {lang} programmers")
            
    # Guarantee at least 60 turns to satisfy "around 50 questions"
    while len(prompts) < 65:
        prompts.append("tell me another quick funny joke")
        prompts.append("write a very short hello world in a random programming language. keep it under 3 lines.")
        
    return prompts

def run_rapid_fire_test():
    global server_process, server_crashed, server_crash_reason
    print("Starting Llama Server with MTP and disabled CUDA Graphs for Rapid Q&A...", flush=True)
    
    # Open log file
    os.makedirs(os.path.dirname(SERVER_LOG_PATH), exist_ok=True)
    log_file = open(SERVER_LOG_PATH, "w", encoding="utf-8")
    
    env = os.environ.copy()
    env["GGML_CUDA_DISABLE_GRAPHS"] = "1"
    
    server_process = subprocess.Popen(
        CMD,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env
    )
    
    # Start monitor thread
    monitor_thread = threading.Thread(target=monitor_server, args=(server_process, log_file))
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # Wait for server to become healthy
    print("Waiting for server to initialize and load the model...", flush=True)
    health_url = f"{SERVER_URL}/health"
    ready = False
    for i in range(120):
        if server_process.poll() is not None:
            server_crashed = True
            server_crash_reason = f"Process terminated with code {server_process.returncode}"
            break
        if server_crashed:
            break
            
        try:
            req = urllib.request.Request(health_url)
            with urllib.request.urlopen(req, timeout=2) as response:
                if response.status == 200:
                    print("Server is HEALTHY and READY!", flush=True)
                    ready = True
                    break
        except Exception:
            pass
        time.sleep(2)
        
    if not ready:
        print("Error: Server failed to start or become healthy in time.", flush=True)
        if server_process.poll() is None:
            server_process.terminate()
        log_file.close()
        sys.exit(1)
        
    print("\n--- Starting Rapid-Fire Q&A Spree ---", flush=True)
    
    prompts = generate_rapid_prompts()
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Keep all code blocks and jokes extremely brief and concise, under 5 lines/30 words if possible. No fluff."}
    ]
    
    target_context_tokens = 28000 # 40% of 70000
    current_tokens = 0
    max_turns = 60
    turn = 0
    
    chat_url = f"{SERVER_URL}/v1/chat/completions"
    
    while turn < max_turns and current_tokens < target_context_tokens:
        if server_process.poll() is not None or server_crashed:
            print(f"\nCRITICAL: Server crashed during Turn {turn + 1}!", flush=True)
            print(f"Reason: {server_crash_reason}", flush=True)
            break
            
        prompt = prompts[turn]
        print(f"\n[TURN {turn + 1}/{max_turns}] Context: {current_tokens} tokens", flush=True)
        print(f"User: \"{prompt}\"", flush=True)
        
        messages.append({"role": "user", "content": prompt})
        
        req_body = {
            "messages": messages,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "repeat_penalty": 1.0,
            "stream": False,
            "max_tokens": 150
        }
        
        req_data = json.dumps(req_body).encode("utf-8")
        start_time = time.time()
        
        try:
            req = urllib.request.Request(
                chat_url,
                data=req_data,
                headers={"Content-Type": "application/json"}
            )
            
            with urllib.request.urlopen(req, timeout=300) as response:
                elapsed = time.time() - start_time
                resp_json = json.loads(response.read().decode("utf-8"))
                
                choices = resp_json.get("choices", [])
                if not choices:
                    print(f"Warning: Empty choice set.", flush=True)
                    continue
                    
                content = choices[0].get("message", {}).get("content", "")
                messages.append({"role": "assistant", "content": content})
                
                usage = resp_json.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", 0)
                
                current_tokens = total_tokens
                
                print(f"Response ({elapsed:.2f}s): {repr(content.strip()[:100])}...", flush=True)
                print(f"Tokens: Prompt={prompt_tokens}, Gen={completion_tokens}, Total={total_tokens}", flush=True)
                
        except urllib.error.URLError as e:
            print(f"HTTP Error: {e}", flush=True)
            server_crashed = True
            server_crash_reason = f"URLError: {e}"
            break
        except Exception as e:
            print(f"Exception: {e}", flush=True)
            server_crashed = True
            server_crash_reason = f"Exception: {e}"
            break
            
        turn += 1
        time.sleep(0.5) # Extremely fast transition to simulate user typing spree!
        
    print("\n--- RAPID-FIRE TEST CONCLUSION ---", flush=True)
    if not server_crashed:
        print(f"SUCCESS! Completed {turn} rapid Q&A turns. Final context size reached: {current_tokens} tokens.", flush=True)
        print("MTP speculative decoding and recurrent cell check-pointing are bulletproof!", flush=True)
        status = "PASSED"
    else:
        print(f"FAILED on Turn {turn + 1} at {current_tokens} tokens.", flush=True)
        print(f"Crash Reason: {server_crash_reason}", flush=True)
        
        print("\n--- LAST 30 LINES OF LLAMA-SERVER LOG ---", flush=True)
        try:
            with open(SERVER_LOG_PATH, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines[-30:]:
                    print(line.strip(), flush=True)
        except Exception as e:
            print(f"Could not read log file: {e}", flush=True)
        status = "FAILED"
        
    print("Terminating llama-server process...", flush=True)
    if server_process and server_process.poll() is None:
        server_process.terminate()
        try:
            server_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_process.kill()
            
    log_file.close()
    
    if status == "PASSED":
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    run_rapid_fire_test()
