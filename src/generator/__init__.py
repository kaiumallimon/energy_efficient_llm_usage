from src.generator.models import GeneratedPrompt, InferenceParams, ModelTier

__all__ = ["AdaptivePromptGenerator", "GeneratedPrompt", "InferenceParams", "ModelTier"]


def __getattr__(name: str):
    if name == "AdaptivePromptGenerator":
        from src.generator.generator import AdaptivePromptGenerator

        return AdaptivePromptGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
