from __future__ import annotations

from src.analyzer.models import ComplexityResult, OptimizationPolicy
from src.decomposer.models import DecomposedPrompt
from src.optimizer.llm_optimizer import LLMPromptOptimizer
from src.optimizer.models import OptimizationResult
from src.optimizer.rules import (
    count_words,
    dedupe_lines,
    normalize_whitespace,
    optimize_query_text,
    trim_context,
)
from src.validator.models import ValidationResult
from src.validator.validator import QualityValidator


class PromptOptimizer:
    """Energy-aware semantic prompt optimizer with LLM compression and quality validation."""

    def __init__(
        self,
        llm_optimizer: LLMPromptOptimizer | None = None,
        validator: QualityValidator | None = None,
        *,
        use_ollama: bool = True,
    ) -> None:
        self.llm_optimizer = llm_optimizer or LLMPromptOptimizer(use_ollama=use_ollama)
        self.validator = validator or QualityValidator(use_ollama=use_ollama)
        self.use_ollama = use_ollama

    def optimize(
        self,
        query: str,
        context: str | None,
        analysis: ComplexityResult,
        decomposed: DecomposedPrompt | None = None,
    ) -> OptimizationResult:
        from src.decomposer.decomposer import PromptDecomposer

        policy = analysis.policy
        original_query = query.strip()
        original_context = context.strip() if context else None
        structured = decomposed or PromptDecomposer(use_ollama=False).decompose(
            original_query,
            original_context,
            analysis,
        )

        optimized_query = original_query
        optimized_context = original_context
        changes: list[str] = []
        validation: ValidationResult | None = None
        optimizer_source = "rules"

        if policy != OptimizationPolicy.MINIMAL:
            llm_query, llm_changes = self.llm_optimizer.optimize_query(
                original_query,
                analysis,
                structured,
            )
            if llm_changes and llm_query != original_query:
                validation = self.validator.validate(original_query, llm_query)
                optimized_query = validation.accepted_query
                changes.extend(llm_changes)
                changes.extend(validation.notes)
                optimizer_source = "llm"
            elif self.use_ollama is False or policy == OptimizationPolicy.AGGRESSIVE:
                optimized_query, rule_changes = self._apply_rule_optimizer(
                    original_query,
                    policy,
                )
                changes.extend(rule_changes)
                optimizer_source = "rules"

        if optimized_context:
            optimized_context, context_changes = self._optimize_context(
                optimized_context,
                original_context,
                policy,
            )
            changes.extend(context_changes)

        return self._build_result(
            original_query,
            optimized_query,
            original_context,
            optimized_context,
            analysis,
            changes,
            validation,
            optimizer_source,
        )

    def _apply_rule_optimizer(
        self,
        query: str,
        policy: OptimizationPolicy,
    ) -> tuple[str, list[str]]:
        aggressive = policy == OptimizationPolicy.AGGRESSIVE
        moderate = policy in {OptimizationPolicy.MODERATE, OptimizationPolicy.AGGRESSIVE}
        return optimize_query_text(query, aggressive=aggressive, moderate=moderate)

    def _optimize_context(
        self,
        context: str,
        original_context: str | None,
        policy: OptimizationPolicy,
    ) -> tuple[str, list[str]]:
        changes: list[str] = []
        optimized = context

        if policy in {OptimizationPolicy.AGGRESSIVE, OptimizationPolicy.CONSERVATIVE}:
            optimized, dedupe_changes = dedupe_lines(optimized)
            changes.extend(dedupe_changes)

        optimized = normalize_whitespace(optimized)
        if optimized != original_context:
            changes.append("Normalized context whitespace.")

        if policy == OptimizationPolicy.AGGRESSIVE:
            optimized, trim_changes = trim_context(optimized, max_words=120)
            changes.extend(trim_changes)

        return optimized, changes

    def _build_result(
        self,
        original_query: str,
        optimized_query: str,
        original_context: str | None,
        optimized_context: str | None,
        analysis: ComplexityResult,
        changes: list[str],
        validation: ValidationResult | None,
        optimizer_source: str,
    ) -> OptimizationResult:
        original_words = count_words(original_query)
        if original_context:
            original_words += count_words(original_context)

        optimized_words = count_words(optimized_query)
        if optimized_context:
            optimized_words += count_words(optimized_context)

        if original_words == 0:
            reduction = 0.0
        else:
            reduction = ((original_words - optimized_words) / original_words) * 100.0

        was_modified = (
            optimized_query != original_query
            or optimized_context != original_context
            or bool(changes)
        )

        if not changes and not was_modified:
            changes = ["No optimization applied for current policy."]

        return OptimizationResult(
            original_query=original_query,
            optimized_query=optimized_query,
            original_context=original_context,
            optimized_context=optimized_context,
            policy=analysis.policy,
            complexity_level=analysis.level,
            changes=changes,
            original_word_count=original_words,
            optimized_word_count=optimized_words,
            word_reduction_percent=max(0.0, reduction),
            was_modified=was_modified,
            validation=validation,
            optimizer_source=optimizer_source,
        )
