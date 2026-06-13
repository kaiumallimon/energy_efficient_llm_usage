from __future__ import annotations

from src.analyzer.models import ComplexityResult, OptimizationPolicy
from src.optimizer.models import OptimizationResult
from src.optimizer.rules import (
    compress_redundancy,
    count_words,
    dedupe_lines,
    normalize_whitespace,
    remove_fillers,
    trim_context,
)


class PromptOptimizer:
    """Rule-based prompt optimizer driven by complexity analysis policy."""

    def optimize(
        self,
        query: str,
        context: str | None,
        analysis: ComplexityResult,
    ) -> OptimizationResult:
        policy = analysis.policy
        original_query = query.strip()
        original_context = context.strip() if context else None

        optimized_query = original_query
        optimized_context = original_context
        changes: list[str] = []

        if policy == OptimizationPolicy.MINIMAL:
            return self._build_result(
                original_query,
                optimized_query,
                original_context,
                optimized_context,
                analysis,
                changes,
            )

        optimized_query = normalize_whitespace(optimized_query)
        if optimized_query != original_query:
            changes.append("Normalized query whitespace.")

        if policy in {OptimizationPolicy.AGGRESSIVE, OptimizationPolicy.MODERATE}:
            optimized_query, filler_changes = remove_fillers(
                optimized_query,
                aggressive=policy == OptimizationPolicy.AGGRESSIVE,
            )
            changes.extend(filler_changes)

        if policy == OptimizationPolicy.AGGRESSIVE:
            optimized_query, redundancy_changes = compress_redundancy(optimized_query)
            changes.extend(redundancy_changes)

        if optimized_context:
            if policy in {OptimizationPolicy.AGGRESSIVE, OptimizationPolicy.CONSERVATIVE}:
                optimized_context, dedupe_changes = dedupe_lines(optimized_context)
                changes.extend(dedupe_changes)

            optimized_context = normalize_whitespace(optimized_context)
            if optimized_context != original_context:
                changes.append("Normalized context whitespace.")

            if policy == OptimizationPolicy.AGGRESSIVE:
                optimized_context, trim_changes = trim_context(optimized_context, max_words=120)
                changes.extend(trim_changes)

        return self._build_result(
            original_query,
            optimized_query,
            original_context,
            optimized_context,
            analysis,
            changes,
        )

    def _build_result(
        self,
        original_query: str,
        optimized_query: str,
        original_context: str | None,
        optimized_context: str | None,
        analysis: ComplexityResult,
        changes: list[str],
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
        )
