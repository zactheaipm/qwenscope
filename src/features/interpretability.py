"""Automated feature interpretation using Claude API.

For each feature, finds top-activating examples and asks Claude
to describe what the feature represents.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch

from src.model.hooks import ActivationCache
from src.sae.model import TopKSAE

logger = logging.getLogger(__name__)


@dataclass
class TokenActivation:
    """A single token window with its feature activation strength.

    Attributes:
        text: The full text this token belongs to.
        token: The individual token string.
        token_position: Position of the token in the tokenized sequence.
        activation: The feature activation value at this position.
        context_window: Surrounding text around the token for readability.
        region: Chat template region this token belongs to (system/user/assistant).
    """

    text: str
    token: str
    token_position: int
    activation: float
    context_window: str
    region: str = ""


@dataclass
class PositionDistribution:
    """Distribution of top-activating tokens across chat template regions.

    If a feature's activations concentrate in the system prompt region,
    it is likely encoding instruction-sensitivity rather than a genuine
    behavioral trait.

    Attributes:
        feature_idx: The SAE feature index.
        total_tokens: Number of top-activating tokens analyzed.
        system_frac: Fraction of tokens in the system prompt region.
        user_frac: Fraction of tokens in the user message region.
        assistant_frac: Fraction of tokens in the assistant response region.
        other_frac: Fraction in template tokens or unclassified regions.
    """

    feature_idx: int
    total_tokens: int
    system_frac: float
    user_frac: float
    assistant_frac: float
    other_frac: float

    @property
    def is_system_dominated(self) -> bool:
        """True if >50% of activations are in the system prompt region."""
        return self.system_frac > 0.5


class AutoInterp:
    """Automated feature interpretation using Claude API.

    For each feature:
    1. Find top-20 activating examples from the training data
    2. Send to Claude with a structured prompt
    3. Get a natural-language description of what the feature represents
    """

    INTERP_PROMPT = """You are an expert in mechanistic interpretability of language models.

I'm going to show you text examples that maximally activate a specific feature
in a Sparse Autoencoder trained on a large language model's residual stream.

Your task: Based on the examples, provide a concise (1-2 sentence) description
of what this feature appears to encode. Focus on the COMMON PATTERN across examples.

Here are the top-activating examples for feature #{feature_idx}:

{examples}

What does this feature encode? Respond with ONLY the description, no preamble."""

    # Default batch size for processing texts through the model.
    DEFAULT_BATCH_SIZE: int = 8

    # Number of tokens of context on each side when building token windows.
    TOKEN_CONTEXT_WINDOW: int = 5

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        layer: int,
        model_name: str = "claude-sonnet-4-20250514",
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        """Initialize the auto-interpreter.

        Args:
            model: The language model (HuggingFace model with model.model.layers).
            tokenizer: The tokenizer for the language model.
            layer: The layer index at which the SAE hooks into the residual stream.
            model_name: Claude model to use for interpretation.
            batch_size: Number of texts to process in each forward-pass batch.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.layer = layer
        self.model_name = model_name
        self.batch_size = batch_size
        self._client = None

    def _get_client(self) -> Any:
        """Lazy-init the Anthropic client."""
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    def interpret_feature(
        self,
        sae: TopKSAE,
        feature_idx: int,
        training_data_sample: list[str],
    ) -> str:
        """Get a natural-language interpretation of one feature.

        Args:
            sae: The trained SAE.
            feature_idx: Index of the feature to interpret.
            training_data_sample: List of text samples to find top activations in.

        Returns:
            Natural-language description of the feature.
        """
        # Find top-activating examples
        top_examples = self._find_top_activating(
            sae, feature_idx, training_data_sample, top_n=20
        )

        if not top_examples:
            return "No activating examples found."

        # Format examples
        examples_text = "\n\n".join(
            f"Example {i+1} (activation: {act:.3f}):\n{text}"
            for i, (text, act) in enumerate(top_examples)
        )

        prompt = self.INTERP_PROMPT.format(
            feature_idx=feature_idx,
            examples=examples_text,
        )

        # Call Claude
        try:
            client = self._get_client()
            response = client.messages.create(
                model=self.model_name,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            description = response.content[0].text.strip()
            logger.info("Feature %d: %s", feature_idx, description)
            return description
        except Exception as e:
            logger.error("Auto-interp failed for feature %d: %s", feature_idx, e)
            return f"Interpretation failed: {e}"

    def interpret_top_features(
        self,
        sae: TopKSAE,
        tas_scores: torch.Tensor,
        training_data_sample: list[str],
        top_k: int = 20,
    ) -> dict[int, str]:
        """Interpret the top-k features by TAS score.

        Args:
            sae: The trained SAE.
            tas_scores: TAS tensor of shape (dict_size,).
            training_data_sample: Text samples for finding activations.
            top_k: Number of top features to interpret.

        Returns:
            Dict mapping feature index to interpretation string.
        """
        abs_tas = tas_scores.abs()
        k = min(top_k, abs_tas.shape[0])
        _, topk_indices = torch.topk(abs_tas, k)

        interpretations: dict[int, str] = {}
        for idx in topk_indices.tolist():
            interpretations[idx] = self.interpret_feature(
                sae, idx, training_data_sample
            )

        return interpretations

    def _get_feature_activations_for_batch(
        self,
        sae: TopKSAE,
        texts: list[str],
        feature_idx: int,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Run a batch of texts through model + SAE and return per-token feature activations.

        Args:
            sae: The trained SAE.
            texts: Batch of text strings.
            feature_idx: Index of the feature to extract.

        Returns:
            List of (feature_activations, attention_mask) tuples, one per text.
            feature_activations has shape (seq_len,) -- the activation of
            feature_idx at each token position.
            attention_mask has shape (seq_len,) -- 1 for real tokens, 0 for padding.
        """
        # Tokenize the batch
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=True,
        )
        device = next(self.model.parameters()).device
        input_ids = encoded["input_ids"].to(device=device)  # (batch, seq_len)
        attention_mask = encoded["attention_mask"].to(device=device)  # (batch, seq_len)

        # Run forward pass with hook to capture residual stream at target layer
        cache = ActivationCache(self.model, layers=[self.layer])
        with torch.no_grad(), cache.active():
            self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Get cached activations: (batch, seq_len, hidden_dim)
        residual = cache.get(self.layer)

        # Move SAE to same device and dtype as residual for encoding
        sae_device = next(sae.parameters()).device
        if sae_device != residual.device:
            residual_for_sae = residual.to(device=sae_device)
        else:
            residual_for_sae = residual

        # Encode through SAE to get sparse features: (batch, seq_len, dict_size)
        with torch.no_grad():
            sparse_features = sae.encode(residual_for_sae)

        # Extract the single feature we care about: (batch, seq_len)
        feature_acts = sparse_features[:, :, feature_idx].cpu()
        attention_mask_cpu = attention_mask.cpu()

        cache.clear()

        results: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i in range(len(texts)):
            results.append((feature_acts[i], attention_mask_cpu[i]))
        return results

    def _find_top_activating(
        self,
        sae: TopKSAE,
        feature_idx: int,
        texts: list[str],
        top_n: int = 20,
    ) -> list[tuple[str, float]]:
        """Find texts that most strongly activate a given feature.

        For each text, tokenizes it, runs it through the model to capture the
        residual stream at the SAE's target layer, encodes through the SAE,
        and records the maximum activation of the target feature across all
        token positions. Returns the top-N texts sorted by max activation.

        Args:
            sae: The trained SAE.
            feature_idx: Feature index to search for.
            texts: Text samples to score.
            top_n: Number of top examples to return.

        Returns:
            List of (text, max_activation) tuples sorted descending by activation.
        """
        if not texts:
            return []

        # Collect (text, max_activation) for every input text
        text_activations: list[tuple[str, float]] = []

        # Process in batches for efficiency
        for batch_start in range(0, len(texts), self.batch_size):
            batch_texts = texts[batch_start : batch_start + self.batch_size]

            batch_results = self._get_feature_activations_for_batch(
                sae, batch_texts, feature_idx
            )

            for text, (feat_acts, mask) in zip(batch_texts, batch_results):
                # Mask out padding positions by setting them to -inf
                masked_acts = feat_acts.clone()
                masked_acts[mask == 0] = float("-inf")

                # Take max activation across all real token positions
                max_act = masked_acts.max().item()

                # Skip texts where the feature never fires (max is 0 or -inf)
                if max_act > 0:
                    text_activations.append((text, max_act))

            if batch_start % (self.batch_size * 10) == 0 and batch_start > 0:
                logger.debug(
                    "Top-activating search: processed %d / %d texts",
                    batch_start,
                    len(texts),
                )

        # Sort by activation descending and return top-N
        text_activations.sort(key=lambda x: x[1], reverse=True)
        n = min(top_n, len(text_activations))

        logger.info(
            "Feature %d: found %d activating texts out of %d total, returning top %d",
            feature_idx,
            len(text_activations),
            len(texts),
            n,
        )
        return text_activations[:n]

    def analyze_position_distribution(
        self,
        sae: TopKSAE,
        feature_indices: list[int],
        chat_texts: list[list[dict[str, Any]]],
        top_n: int = 50,
    ) -> list[PositionDistribution]:
        """Classify top-activating token positions by chat template region.

        For each feature, finds the top-N activating tokens and determines
        whether they fall in the system prompt, user message, or assistant
        response region. Features dominated by system-prompt activations
        are likely encoding instruction-sensitivity rather than behavioral
        traits.

        Args:
            sae: The trained SAE.
            feature_indices: Feature indices to analyze.
            chat_texts: List of chat message lists (each is a
                ``list[dict]`` with "role" and "content" keys).
            top_n: Number of top-activating tokens per feature to analyze.

        Returns:
            List of PositionDistribution objects, one per feature.
        """
        results: list[PositionDistribution] = []

        for feat_idx in feature_indices:
            region_counts = {"system": 0, "user": 0, "assistant": 0, "other": 0}
            total = 0

            for messages in chat_texts:
                # Build the full text using the chat template so we can
                # map token positions back to roles.
                try:
                    full_text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except Exception:
                    continue

                # Compute region boundaries in character space
                region_spans = self._compute_region_spans(messages, full_text)

                # Tokenize the full text
                encoded = self.tokenizer(
                    full_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048,
                )
                device = next(self.model.parameters()).device
                input_ids = encoded["input_ids"].to(device)

                # Get feature activations for this text
                cache = ActivationCache(self.model, layers=[self.layer])
                with torch.no_grad(), cache.active():
                    self.model(input_ids=input_ids)
                residual = cache.get(self.layer)
                sae_device = next(sae.parameters()).device
                residual_for_sae = residual.to(device=sae_device)
                with torch.no_grad():
                    sparse_features = sae.encode(residual_for_sae)
                feat_acts = sparse_features[0, :, feat_idx].cpu()  # (seq_len,)
                cache.clear()

                # Map each activating position to a region
                for pos in range(feat_acts.shape[0]):
                    if feat_acts[pos].item() <= 0:
                        continue
                    # Decode token to find its character offset
                    # Use offset_mapping if available, otherwise estimate
                    char_offset = self._estimate_char_offset(
                        encoded, pos, full_text
                    )
                    region = self._classify_position(char_offset, region_spans)
                    region_counts[region] += 1
                    total += 1

            if total == 0:
                results.append(PositionDistribution(
                    feature_idx=feat_idx,
                    total_tokens=0,
                    system_frac=0.0,
                    user_frac=0.0,
                    assistant_frac=0.0,
                    other_frac=1.0,
                ))
            else:
                results.append(PositionDistribution(
                    feature_idx=feat_idx,
                    total_tokens=total,
                    system_frac=region_counts["system"] / total,
                    user_frac=region_counts["user"] / total,
                    assistant_frac=region_counts["assistant"] / total,
                    other_frac=region_counts["other"] / total,
                ))

            logger.info(
                "Feature %d position distribution: system=%.2f user=%.2f "
                "assistant=%.2f other=%.2f (n=%d)",
                feat_idx,
                results[-1].system_frac,
                results[-1].user_frac,
                results[-1].assistant_frac,
                results[-1].other_frac,
                total,
            )

        return results

    def _compute_region_spans(
        self,
        messages: list[dict[str, Any]],
        full_text: str,
    ) -> list[tuple[str, int, int]]:
        """Compute character-level spans for each message region in the template.

        Args:
            messages: Chat messages with "role" and "content".
            full_text: The full text after applying the chat template.

        Returns:
            List of (role, start_char, end_char) tuples.
        """
        spans: list[tuple[str, int, int]] = []
        search_start = 0
        for msg in messages:
            content = msg.get("content", "")
            if not content:
                continue
            idx = full_text.find(content, search_start)
            if idx >= 0:
                spans.append((msg["role"], idx, idx + len(content)))
                search_start = idx + len(content)
        return spans

    @staticmethod
    def _classify_position(
        char_offset: int,
        region_spans: list[tuple[str, int, int]],
    ) -> str:
        """Classify a character offset into a chat region.

        Args:
            char_offset: Character position in the full text.
            region_spans: List of (role, start, end) tuples.

        Returns:
            One of "system", "user", "assistant", "other".
        """
        for role, start, end in region_spans:
            if start <= char_offset < end:
                return role
        return "other"

    def _estimate_char_offset(
        self,
        encoded: dict[str, torch.Tensor],
        token_pos: int,
        full_text: str,
    ) -> int:
        """Estimate the character offset of a token position.

        Uses prefix decoding: decode tokens 0..pos and measure the length.

        Args:
            encoded: Tokenizer output with "input_ids".
            token_pos: Token position index.
            full_text: The original full text.

        Returns:
            Estimated character offset.
        """
        prefix_ids = encoded["input_ids"][0, : token_pos + 1].tolist()
        prefix_text = self.tokenizer.decode(prefix_ids, skip_special_tokens=False)
        return len(prefix_text) - 1

    def validate_cross_corpus(
        self,
        sae: TopKSAE,
        feature_indices: list[int],
        instructional_texts: list[str],
        non_instructional_texts: list[str],
        top_n: int = 20,
    ) -> dict[int, dict[str, float]]:
        """Check whether features activate on non-instructional text too.

        Features that are truly behavioral should activate primarily on
        instructional/agentic text. Features that also activate strongly
        on non-instructional text (e.g., Wikipedia, code) may be encoding
        general linguistic patterns rather than behavioral traits.

        Args:
            sae: The trained SAE.
            feature_indices: Features to validate.
            instructional_texts: Instruction-following text samples.
            non_instructional_texts: Non-instructional text samples
                (e.g., Wikipedia, plain code, novels).
            top_n: Number of top-activating examples to compare.

        Returns:
            Dict mapping feature_idx to {
                "instructional_max_act": float,
                "non_instructional_max_act": float,
                "specificity_ratio": float,  # instructional / non_instructional
                "is_instruction_specific": bool,  # ratio > 2.0
            }.
        """
        results: dict[int, dict[str, float]] = {}

        for feat_idx in feature_indices:
            # Find top activations on instructional text
            inst_top = self._find_top_activating(
                sae, feat_idx, instructional_texts, top_n=top_n
            )
            inst_max = inst_top[0][1] if inst_top else 0.0

            # Find top activations on non-instructional text
            non_inst_top = self._find_top_activating(
                sae, feat_idx, non_instructional_texts, top_n=top_n
            )
            non_inst_max = non_inst_top[0][1] if non_inst_top else 0.0

            specificity = inst_max / max(non_inst_max, 1e-8)

            results[feat_idx] = {
                "instructional_max_act": inst_max,
                "non_instructional_max_act": non_inst_max,
                "specificity_ratio": specificity,
                "is_instruction_specific": specificity > 2.0,
            }

            logger.info(
                "Feature %d cross-corpus: inst_max=%.3f, non_inst_max=%.3f, "
                "specificity=%.2f",
                feat_idx, inst_max, non_inst_max, specificity,
            )

        return results

    def interpret_feature_batch1(
        self,
        sae: TopKSAE,
        feature_idx: int,
        training_data_sample: list[str],
    ) -> str:
        """Re-run interpretation with batch_size=1 to eliminate batching artifacts.

        Padding in batched processing can introduce spurious activations,
        particularly for features that fire on padding tokens. This method
        processes each text individually to verify the interpretation.

        Args:
            sae: The trained SAE.
            feature_idx: Index of the feature to interpret.
            training_data_sample: List of text samples.

        Returns:
            Natural-language description of the feature.
        """
        original_batch_size = self.batch_size
        self.batch_size = 1
        try:
            return self.interpret_feature(sae, feature_idx, training_data_sample)
        finally:
            self.batch_size = original_batch_size

    def _find_top_activating_tokens(
        self,
        sae: TopKSAE,
        feature_idx: int,
        texts: list[str],
        top_n: int = 20,
        context_tokens: int | None = None,
    ) -> list[TokenActivation]:
        """Find the specific tokens that most strongly activate a given feature.

        Unlike ``_find_top_activating`` which returns whole texts, this method
        returns individual token positions with surrounding context windows,
        enabling more granular interpretability analysis.

        Args:
            sae: The trained SAE.
            feature_idx: Feature index to search for.
            texts: Text samples to scan.
            top_n: Number of top token activations to return.
            context_tokens: Number of tokens on each side of the activating
                token to include in the context window. Defaults to
                ``TOKEN_CONTEXT_WINDOW``.

        Returns:
            List of ``TokenActivation`` objects sorted descending by activation.
        """
        if not texts:
            return []

        if context_tokens is None:
            context_tokens = self.TOKEN_CONTEXT_WINDOW

        # Collect all candidate token activations across all texts
        all_token_acts: list[TokenActivation] = []

        for batch_start in range(0, len(texts), self.batch_size):
            batch_texts = texts[batch_start : batch_start + self.batch_size]

            # Tokenize again for token-level decoding (need the token IDs)
            encoded = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                padding=True,
            )
            batch_input_ids = encoded["input_ids"]  # (batch, seq_len)

            batch_results = self._get_feature_activations_for_batch(
                sae, batch_texts, feature_idx
            )

            for text_idx, (text, (feat_acts, mask)) in enumerate(
                zip(batch_texts, batch_results)
            ):
                input_ids = batch_input_ids[text_idx]  # (seq_len,)
                seq_len = int(mask.sum().item())  # number of real tokens

                for pos in range(seq_len):
                    act_value = feat_acts[pos].item()
                    if act_value <= 0:
                        continue

                    # Decode the activating token
                    token_str = self.tokenizer.decode(
                        [input_ids[pos].item()], skip_special_tokens=False
                    )

                    # Build context window around the activating token
                    ctx_start = max(0, pos - context_tokens)
                    ctx_end = min(seq_len, pos + context_tokens + 1)
                    context_ids = input_ids[ctx_start:ctx_end].tolist()
                    context_str = self.tokenizer.decode(
                        context_ids, skip_special_tokens=False
                    )

                    all_token_acts.append(
                        TokenActivation(
                            text=text,
                            token=token_str,
                            token_position=pos,
                            activation=act_value,
                            context_window=context_str,
                        )
                    )

            if batch_start % (self.batch_size * 10) == 0 and batch_start > 0:
                logger.debug(
                    "Top-activating token search: processed %d / %d texts",
                    batch_start,
                    len(texts),
                )

        # Sort by activation descending and return top-N
        all_token_acts.sort(key=lambda x: x.activation, reverse=True)
        n = min(top_n, len(all_token_acts))

        logger.info(
            "Feature %d: found %d activating tokens across %d texts, returning top %d",
            feature_idx,
            len(all_token_acts),
            len(texts),
            n,
        )
        return all_token_acts[:n]
