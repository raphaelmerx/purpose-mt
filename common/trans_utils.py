import json
from hashlib import md5
from typing import List, Optional, Sequence, Callable
from google.cloud import translate_v2 as translate
import random

import diskcache
from dotenv import load_dotenv
load_dotenv()

import datasets
from pydantic import BaseModel
from litellm import completion
from sacrebleu import corpus_chrf

try:  # optional: only used when running under a wandb sweep (see pyproject [train])
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


cache_dir = "cache/llm_responses"
cache = diskcache.Cache(cache_dir)


class Wmt24ppSegment(BaseModel):
    lang: str
    domain: str
    document_id: str
    segment_id: int
    is_bad_source: bool
    source: str
    target: Optional[str]
    _pred: str

    @classmethod
    def load(cls, lang) -> Sequence["Wmt24ppSegment"]:
        if lang == "tet":
            with open("datasets/wmt24pp_en-tet_gtrans.jsonl", "r", encoding="utf-8") as f:
                rows = [cls.model_validate_json(line) for line in f]
            return rows
        ds = datasets.load_dataset("google/wmt24pp", lang, split="train")
        rows = [row for row in ds]
        for row in rows:
            row["lang"] = row.pop("lp")
        return [cls.model_validate(row) for row in rows]


def cached_completion(*args, **kwargs) -> str:
    """
    A wrapper around litellm's completion function to cache results.

    Cache uses diskcache, which is thread-safe and persistent.
    Key will be the md5 hash of the messages + model.
    """

    messages = kwargs.get('messages', [])
    response_format = kwargs.get('response_format', {})
    # if it's a Pydantic model, get its model_json_schema
    if hasattr(response_format, 'model_json_schema'):
        response_format = response_format.model_json_schema()
    hash_key = json.dumps(messages, sort_keys=True) + kwargs.get('model', '') + json.dumps(response_format, sort_keys=True)
    hash_key = md5(hash_key.encode('utf-8')).hexdigest()

    if hash_key in cache and cache[hash_key]:
        return cache[hash_key]

    response = completion(*args, **kwargs)
    result = response.choices[0].message.content
    if not result:
        raise ValueError(f"Completion failed for messages: {messages} with response {response}")

    if result:
        cache[hash_key] = result

    return result

def gtranslate(text: str, target_lang: str = 'en') -> str:
    """
    Translate text to the target language using Google Translate
    """
    client = translate.Client()
    translation = client.translate(text, target_language=target_lang, format_="text")
    return translation['translatedText']


def test_model(
    predict_fn: Callable[[List[str]], List[str]],
    lang: str = "tet",
    sample_size: int = 100
) -> List[Wmt24ppSegment]:
    """
    Test a model on WMT24++ dataset.
    
    Args:
        predict_fn: A function that takes a list of source texts and returns a list of predictions
        lang: The target language to test on
        sample_size: Number of random samples to print
    """
    segments = Wmt24ppSegment.load(lang)
    print(f"\nLoaded {len(segments)} WMT24pp segments for {lang}")

    print("\nRunning inference on WMT24++ English sources...")
    
    # Generate predictions using provided function
    sources = [s.source for s in segments]
    predictions = predict_fn(sources)
    
    # Store predictions in segments
    for segment, pred in zip(segments, predictions):
        segment._pred = pred
    
    # Print a random sample
    actual_sample_size = min(sample_size, len(segments))
    sample_segments = random.sample(segments, actual_sample_size)
    print(f"\n--- Random sample of {actual_sample_size} translations ---")
    for segment in sample_segments:
        print(f"EN: {segment.source}")
        print(f"REF: {segment.target}")
        print(f"PRED: {segment._pred}")
        print()
    
    # Calculate overall ChrF++ score
    references = [s.target if s.target else "" for s in segments]
    chrf = corpus_chrf(predictions, [references], word_order=2)
    print(f"\n=== WMT24++ ChrF++ Score: {chrf.score:.2f} ===")
    
    if wandb is not None and wandb.run is not None:
        wandb.log({"test/chrf": chrf.score})
    
    # Calculate domain-level metrics if requested
    unique_domains = set(segment.domain for segment in segments)
    for domain in unique_domains:
        domain_segments = [s for s in segments if s.domain == domain]
        domain_predictions = [s._pred for s in domain_segments]
        domain_references = [s.target if s.target else "" for s in domain_segments]
        domain_chrf = corpus_chrf(domain_predictions, [domain_references], word_order=2)
        print(f"Domain: {domain}, Samples: {len(domain_segments)}, ChrF++: {domain_chrf.score:.2f}")
        if wandb is not None and wandb.run is not None:
            wandb.log({f"test/chrf_{domain}": domain_chrf.score})

    return list(segments)