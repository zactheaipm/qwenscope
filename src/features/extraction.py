"""Run contrastive pairs through model + SAE to extract feature activations."""

from __future__ import annotations

import logging
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict

from src.data.contrastive import BehavioralTrait, ContrastivePair
from src.model.hooks import ActivationCache
from src.sae.model import TopKSAE

logger = logging.getLogger(__name__)

PoolingStrategy = Literal["mean", "max", "last_n", "last_token"]

# Number of trailing non-padding tokens used by the "last_n" strategy.
_LAST_N_TOKENS: int = 32


class FeaturePairResult(BaseModel):
    """Feature activations for one contrastive pair at one SAE.

    Stores pooled feature activations across sequence positions using the
    strategy specified by ``pooling_strategy``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    pair_id: str
    sae_id: str
    pooling_strategy: PoolingStrategy
    features_high_mean: list[float]  # (dict_size,) — pooled activation for HIGH version
    features_low_mean: list[float]   # (dict_size,) — pooled activation for LOW version
    target_sub_behaviors: list[str] = []  # Sub-behaviors this pair targets


class FeatureExtractionResults(BaseModel):
    """Results of feature extraction for all pairs and SAEs."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    trait: BehavioralTrait
    results: dict[str, list[FeaturePairResult]]  # sae_id → list of pair results


class FeatureExtractor:
    """Runs contrastive pairs through model and extracts SAE features.

    For each contrastive pair:
    1. Tokenize both HIGH and LOW versions with tool schemas
    2. Run forward pass through model with hooks active
    3. Extract activations at all 6 SAE positions
    4. Decompose through each SAE
    5. Return feature activations for both versions
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        sae_dict: dict[str, TopKSAE],
        layer_map: dict[str, int],
        device: str = "cuda",
        pooling_strategy: PoolingStrategy = "last_token",
    ) -> None:
        """Initialize the feature extractor.

        Args:
            model: The language model.
            tokenizer: The tokenizer.
            sae_dict: Dict mapping sae_id to trained SAE.
            layer_map: Dict mapping sae_id to layer index.
            device: Computation device.
            pooling_strategy: How to aggregate feature activations across
                sequence positions. One of ``"mean"`` (average over all
                non-padding positions), ``"max"`` (element-wise max over
                non-padding positions), ``"last_n"`` (mean over the last 32
                non-padding tokens), or ``"last_token"`` (final non-padding
                token only).  Defaults to ``"last_token"`` because ``"mean"``
                has a sequence-length confound: different system prompt lengths
                between HIGH and LOW variants dilute decision-point signals
                unequally, creating a systematic bias correlated with the
                trait manipulation.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.sae_dict = sae_dict
        self.layer_map = layer_map
        self.device = device
        self.pooling_strategy: PoolingStrategy = pooling_strategy
        # Cache each SAE's dtype to avoid repeated introspection in the hot loop.
        # Model outputs are typically BF16 but SAE weights may be FP32.
        self._sae_dtypes: dict[str, torch.dtype] = {
            sae_id: next(sae.parameters()).dtype
            for sae_id, sae in sae_dict.items()
        }

    def _tokenize_messages(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Tokenize messages using the chat template.

        Args:
            messages: Chat messages.
            tools: Optional tool schemas to include.

        Returns:
            Dict with 'input_ids' and 'attention_mask' tensors.
        """
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tools=tools if tools else None,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except Exception as e:
            # Fallback without tools if template doesn't support them
            logger.debug("Chat template with tools failed, falling back: %s", e)
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=True,
        )
        return {k: v.to(self.device) for k, v in encoded.items()}

    def _pool_features(
        self,
        features: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pool sparse SAE features across the sequence dimension.

        Args:
            features: SAE-encoded feature activations of shape
                ``(1, seq_len, dict_size)``.
            attention_mask: Attention mask of shape ``(1, seq_len)`` where 1
                indicates a real (non-padding) token.

        Returns:
            Pooled feature tensor of shape ``(1, dict_size)``.
        """
        # mask: (1, seq_len, 1) — broadcast over dict_size
        mask = attention_mask.unsqueeze(-1).float()

        if self.pooling_strategy == "mean":
            pooled = (features * mask).sum(dim=1) / mask.sum(dim=1)  # (1, dict_size)

        elif self.pooling_strategy == "max":
            # Set padded positions to -inf so they never win the max
            fill_value = torch.finfo(features.dtype).min
            masked_features = features.masked_fill(mask == 0, fill_value)
            pooled = masked_features.max(dim=1).values  # (1, dict_size)

        elif self.pooling_strategy == "last_n":
            n = _LAST_N_TOKENS
            # Determine the index of the last non-padding token per batch row
            # lengths: (1,) — number of real tokens in the sequence
            lengths = attention_mask.sum(dim=1)  # (1,)
            batch_pooled = []
            for b in range(features.size(0)):
                seq_len_b = int(lengths[b].item())
                start = max(seq_len_b - n, 0)
                window = features[b, start:seq_len_b, :]  # (window_len, dict_size)
                batch_pooled.append(window.mean(dim=0))  # (dict_size,)
            pooled = torch.stack(batch_pooled, dim=0)  # (1, dict_size)

        elif self.pooling_strategy == "last_token":
            lengths = attention_mask.sum(dim=1)  # (1,)
            batch_pooled = []
            for b in range(features.size(0)):
                last_idx = int(lengths[b].item()) - 1
                batch_pooled.append(features[b, last_idx, :])  # (dict_size,)
            pooled = torch.stack(batch_pooled, dim=0)  # (1, dict_size)

        else:
            raise ValueError(
                f"Unknown pooling strategy: {self.pooling_strategy!r}. "
                f"Expected one of 'mean', 'max', 'last_n', 'last_token'."
            )

        return pooled

    def extract_pair(
        self, pair: ContrastivePair
    ) -> dict[str, FeaturePairResult]:
        """Extract features for one contrastive pair across all SAEs.

        Activations at each SAE hook point are decomposed into sparse features
        and then pooled across sequence positions using the strategy configured
        on this extractor (see ``pooling_strategy``).

        Args:
            pair: The contrastive pair.

        Returns:
            Dict keyed by sae_id, each containing FeaturePairResult.
        """
        layers = list(set(self.layer_map.values()))
        cache = ActivationCache(self.model, layers=layers)

        results: dict[str, FeaturePairResult] = {}

        with torch.no_grad():
            # Process HIGH version
            inputs_high = self._tokenize_messages(pair.messages_high, pair.tools)
            with cache.active():
                self.model(**inputs_high)
            high_acts = {layer: cache.get(layer).clone() for layer in layers}
            cache.clear()

            # Process LOW version
            inputs_low = self._tokenize_messages(pair.messages_low, pair.tools)
            with cache.active():
                self.model(**inputs_low)
            low_acts = {layer: cache.get(layer).clone() for layer in layers}
            cache.clear()

            # Decompose through each SAE
            for sae_id, sae in self.sae_dict.items():
                layer = self.layer_map[sae_id]

                # Cast to SAE dtype before encoding — model outputs are typically
                # BF16 but SAE weights may be FP32, causing a RuntimeError.
                sae_dtype = self._sae_dtypes[sae_id]
                features_high = sae.encode(high_acts[layer].to(dtype=sae_dtype))  # (1, seq_len, dict_size)
                features_low = sae.encode(low_acts[layer].to(dtype=sae_dtype))    # (1, seq_len, dict_size)

                # Pool across sequence positions using the configured strategy
                features_high_pooled = self._pool_features(
                    features_high, inputs_high["attention_mask"],
                )  # (1, dict_size)
                features_low_pooled = self._pool_features(
                    features_low, inputs_low["attention_mask"],
                )  # (1, dict_size)

                results[sae_id] = FeaturePairResult(
                    pair_id=pair.id,
                    sae_id=sae_id,
                    pooling_strategy=self.pooling_strategy,
                    features_high_mean=features_high_pooled.squeeze(0).cpu().tolist(),
                    features_low_mean=features_low_pooled.squeeze(0).cpu().tolist(),
                    target_sub_behaviors=pair.target_sub_behaviors,
                )

        return results

    def extract_all(
        self,
        pairs: list[ContrastivePair],
        trait: BehavioralTrait,
    ) -> FeatureExtractionResults:
        """Extract features for all contrastive pairs. Batches for efficiency.

        Args:
            pairs: List of contrastive pairs for one trait.
            trait: The behavioral trait these pairs test.

        Returns:
            FeatureExtractionResults containing all pair results per SAE.
        """
        all_results: dict[str, list[FeaturePairResult]] = {
            sae_id: [] for sae_id in self.sae_dict
        }

        for i, pair in enumerate(pairs):
            if (i + 1) % 10 == 0:
                logger.info("Extracting features: %d / %d pairs", i + 1, len(pairs))

            pair_results = self.extract_pair(pair)
            for sae_id, result in pair_results.items():
                all_results[sae_id].append(result)

        logger.info(
            "Feature extraction complete: %d pairs × %d SAEs",
            len(pairs),
            len(self.sae_dict),
        )
        return FeatureExtractionResults(trait=trait, results=all_results)

    def compute_mean_activations(
        self,
        pairs: list[ContrastivePair],
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        """Compute mean raw hidden-state activations for HIGH and LOW pairs.

        Used for the mean-diff steering baseline (activation addition without
        SAE decomposition). Returns the average hidden-state vector per layer
        across all contrastive pairs, using the same pooling strategy as
        feature extraction.

        Args:
            pairs: List of contrastive pairs.

        Returns:
            Tuple of (mean_high, mean_low), each a dict mapping layer index
            to a (hidden_dim,) tensor of mean pooled activations.
        """
        layers = list(set(self.layer_map.values()))
        cache = ActivationCache(self.model, layers=layers)

        # Accumulators: layer -> sum of pooled activations
        high_sums: dict[int, torch.Tensor] = {}
        low_sums: dict[int, torch.Tensor] = {}
        n_pairs = 0

        with torch.no_grad():
            for i, pair in enumerate(pairs):
                if (i + 1) % 20 == 0:
                    logger.info(
                        "Mean activations: %d / %d pairs", i + 1, len(pairs)
                    )

                # Process HIGH version
                inputs_high = self._tokenize_messages(pair.messages_high, pair.tools)
                with cache.active():
                    self.model(**inputs_high)
                for layer in layers:
                    act = cache.get(layer)  # (1, seq_len, hidden_dim)
                    pooled = self._pool_features(
                        act, inputs_high["attention_mask"]
                    ).squeeze(0)  # (hidden_dim,)
                    if layer not in high_sums:
                        high_sums[layer] = torch.zeros_like(pooled)
                    high_sums[layer] += pooled
                cache.clear()

                # Process LOW version
                inputs_low = self._tokenize_messages(pair.messages_low, pair.tools)
                with cache.active():
                    self.model(**inputs_low)
                for layer in layers:
                    act = cache.get(layer)
                    pooled = self._pool_features(
                        act, inputs_low["attention_mask"]
                    ).squeeze(0)
                    if layer not in low_sums:
                        low_sums[layer] = torch.zeros_like(pooled)
                    low_sums[layer] += pooled
                cache.clear()

                n_pairs += 1

        # Average
        mean_high = {layer: s / n_pairs for layer, s in high_sums.items()}
        mean_low = {layer: s / n_pairs for layer, s in low_sums.items()}

        logger.info(
            "Mean activations computed: %d pairs × %d layers", n_pairs, len(layers)
        )
        return mean_high, mean_low
