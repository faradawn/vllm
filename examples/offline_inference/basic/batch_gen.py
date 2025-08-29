from vllm import LLM, SamplingParams

# prompts = [
#     "Hello, my name is",
#     "The president of the United States is",
#     "The capital of France is",
#     "The future of AI is",
# ]


prefix = "You are a helpful assistant. " * 100
prompts = [prefix + f"Question {i}: compute {i}+{i} and explain." for i in range(10)]


sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
llm = LLM(model="facebook/opt-125m", enforce_eager=True)
import time

start_time = time.time()

outputs = llm.generate(prompts, sampling_params)
end_time = time.time()
for output in outputs:
    print(f"Generated text: {output.outputs[0].text!r}")

print(f"=== Total time / numer of prompts = {end_time - start_time} / {len(prompts)}")