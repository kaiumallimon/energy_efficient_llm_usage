from __future__ import annotations

from src.analyzer.models import ComplexityResult
from src.generator.models import GeneratedPrompt
from src.generator.templates import (
    build_full_prompt,
    build_messages,
    build_system_prompt,
    build_template_id,
    build_user_prompt,
    select_model_tier,
)
from src.optimizer.models import OptimizationResult


class AdaptivePromptGenerator:
    """Assembles the final LLM-ready prompt from analysis and optimization results."""

    def generate(
        self,
        analysis: ComplexityResult,
        optimization: OptimizationResult,
    ) -> GeneratedPrompt:
        system_prompt, notes = build_system_prompt(analysis)
        user_prompt, sections = build_user_prompt(optimization, analysis)
        messages = build_messages(system_prompt, user_prompt)
        full_prompt = build_full_prompt(system_prompt, user_prompt)

        return GeneratedPrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            context=optimization.optimized_context,
            messages=messages,
            full_prompt=full_prompt,
            model_tier=select_model_tier(analysis),
            template_id=build_template_id(analysis),
            task_type=analysis.task_type,
            complexity_level=analysis.level,
            policy=analysis.policy,
            sections=sections,
            notes=notes,
        )
