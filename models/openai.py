import time
import tiktoken
from copy import deepcopy
from .base import LLMBase
import openai


class ChatGPT(LLMBase):
    def __init__(
        self,
        model_path=None,
        max_attempts=100,
        max_tokens=2048,
        temperature=0,
    ):

        self.client = openai.Client()
        self.max_attempts = max_attempts
        self.delay_seconds = 1
        self.model = model_path.replace("openai/", "")
        self.parameters = {"max_tokens": max_tokens, "temperature": temperature}
        self.num_tokens = 0

    def query(self, prompt):
        pred = self.chat_query(prompt)

        return pred

    def chat_query(self, prompt, messages=None):

        n_attempt = 0
        params = deepcopy(self.parameters)

        if messages is None:
            messages = [{"role": "user", "content": prompt}]

        print("messages", messages)
        while n_attempt < self.max_attempts:
            try:
                completion = self.client.chat.completions.create(
                    model=self.model, messages=messages, **params
                )
                response = completion.choices[0].message.content
                return response
            except Exception as e:
                # Catch any exception that might occur and print an error message
                print(f"An error occurred: {e}, retry {n_attempt}")
                n_attempt += 1
                time.sleep(self.delay_seconds * n_attempt)

        if n_attempt == self.max_attempts:
            print("Max number of attempts reached")
            return ""

        return ""
