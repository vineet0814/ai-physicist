import json, re
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import MODEL_NAME, MAX_NEW_TOKENS, TURN_LIMIT

class CentralLLM:
    def __init__(self, tools, model_path= "dpo_training/best_model"):
        """
        Initialize the CentralLLM with a set of tools and load the model.
        Args:
            tools (list): List of tool functions to be used by the model.
            model_path (str): Path to the pre-trained model.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
        self.tools = tools
        self.memory = []  # Track previous turn summaries/errors

    def try_parse_tool_calls(self, content):
        tool_calls = []
        offset = 0
        for i, m in enumerate(re.finditer(r"<tool_call>\n(.+)?\n</tool_call>", content)):
            if i == 0: offset = m.start()
            try:
                func = json.loads(m.group(1))
                if isinstance(func["arguments"], str):
                    func["arguments"] = json.loads(func["arguments"])
                tool_calls.append({"type": "function", "function": func})
            except: continue
        if tool_calls:
            c = content[:offset] if offset > 0 else ""
            return {"role": "assistant", "content": c, "tool_calls": tool_calls}
        return {"role": "assistant", "content": content.strip()}

    def call_tool(self, fn_name, fn_args):
        for t in self.tools:
            if t.__name__ == fn_name:
                try:
                    return json.dumps(t(**fn_args))
                except Exception as e:
                    return json.dumps({"error": str(e)})
        return json.dumps({"error": "Tool not found"})

    def run_dialogue(self, messages):
        tokenizer, model = self.tokenizer, self.model
        from tools import TOOLS

        def render():
            return tokenizer.apply_chat_template(messages, 
                                                 tools=TOOLS, 
                                                 add_generation_prompt=True, 
                                                 tokenize=False,
                                                  parallel_function_calls=True)

        for turn in range(TURN_LIMIT):
            if self.memory:
                summary = "; ".join(self.memory[-3:])
                messages.append({"role": "system", "content": f"Feedback from prior turns: {summary}. Use this feedback to refine your evaluation."})

            text = render()
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
            output_text = tokenizer.batch_decode(outputs)[0][len(text):]
            tool_msg = self.try_parse_tool_calls(output_text)
            messages.append(tool_msg)

            feedback_log = []
            if tool_calls := tool_msg.get("tool_calls"):
                for call in tool_calls:
                    fn = call["function"]
                    result = self.call_tool(fn["name"], fn["arguments"])
                    messages.append({"role": "tool", "name": fn["name"], "content": result})
                    if "error" in result:
                        feedback_log.append(f"Error in tool {fn['name']}: {result}")

            text = render()
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
            final_text = tokenizer.batch_decode(outputs)[0][len(text):]
            next_msg = self.try_parse_tool_calls(final_text)
            messages.append(next_msg)

            if feedback_log:
                summary_line = f"Turn {turn+1} feedback: {', '.join(feedback_log)}"
                self.memory.append(summary_line)

            if "done" in next_msg["content"].lower():
                break

        return messages
