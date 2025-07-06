import logging
from typing import Optional, Dict, Any

class LLMAgentInterface:
    def __init__(self, config, model_manager):
        self.config = config
        self.model_manager = model_manager
        self.logger = logging.getLogger("LLMAgentInterface")
        self.history = []

    async def generate(
        self,
        prompt: str,
        context: Optional[Dict] = None,
        backend: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """
        Unified method to generate completions or chat with any supported backend.
        - backend: 'ollama' (default), 'openai' (future), etc
        - model: explicit model or best-per-task
        """
        backend = backend or "ollama"
        model = model or self.model_manager.get_optimal_model(kwargs.get('task_type', 'general'))
        self.logger.info(f"LLM request: backend={backend} model={model} prompt={prompt[:80]}...")
        result = None

        if backend == "ollama":
            from langchain_community.chat_models import ChatOllama
            from langchain_core.messages import HumanMessage
            llm = ChatOllama(model=model, temperature=temperature)
            response = await llm.agenerate([[HumanMessage(content=prompt)]])
            result = response.generations[0][0].text
        elif backend == "openai":
            # OpenAI integration can be added here if desired in the future
            raise NotImplementedError("OpenAI backend not yet implemented.")
        else:
            raise ValueError(f"Unknown backend: {backend}")

        entry = {"prompt": prompt, "response": result, "backend": backend, "model": model}
        self.history.append(entry)
        self.logger.info(f"LLM response: {result[:100]}...")
        return result

    def get_history(self) -> Any:
        return self.history

