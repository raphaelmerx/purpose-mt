"""Prompt construction for the controlled-MT benchmarks (CoCoA-MT, MT-GenEval).

System/user message shapes mirror ``common/wmt24pp_utils.py`` so that results are directly
comparable with the WMT24++ baseline and instruction conditions used in the main paper.

CoCoA-MT conditions
-------------------
``C0_zeroshot``   no control, no examples. The purest measure of the model's default register,
                  and the setting prior work on CoCoA-MT uses.
``C0_3shot``      no control, with the same 3 fixed few-shot examples used by the WMT24++
                  baseline. This is the prompt-matched predictor for the mechanism test in
                  ``controlled/correlate_wmt24pp.py``: the WMT24++ baseline is 3-shot, so
                  comparing it against a zero-shot default-register rate would confound prompt
                  format with model default.
``C1_tag_*``      categorical control ("Use formal/informal register"), i.e. the control that
                  prior work on controllable MT uses. Expected near ceiling at 27B.
``C2_brief_*``    a BOUQuET-style brief that names the register alongside audience and domain.
``C3_scenario_*`` a free-text scenario that *implies* a register without ever naming it. The
                  source is held fixed and only the described social relationship varies, so the
                  gold label is the scenario's polarity. This tests whether the model can infer
                  the appropriate adaptation rather than execute a stated category.

MT-GenEval conditions
---------------------
``G0``/``G1``     counterfactual subset, without/with a brief stating the referent's gender.
``X0``-``X3``     contextual subset, mapping onto the paper's own comparison: sentence alone,
                  paragraph context (P1 markers), explicit brief, and self-instruction.
"""
from dataclasses import dataclass
from typing import List, Optional

from translate.wmt24pp_vllm import FEWSHOT_EXAMPLES_BY_LANG


@dataclass(frozen=True)
class CocoaCondition:
    name: str
    target_register: Optional[str]   # "formal" | "informal" | None (no control)
    style: str                       # "none" | "tag" | "brief" | "scenario"
    few_shot: bool = False


COCOA_CONDITIONS: List[CocoaCondition] = [
    CocoaCondition("C0_zeroshot", None, "none", few_shot=False),
    CocoaCondition("C0_3shot", None, "none", few_shot=True),
    CocoaCondition("C1_tag_formal", "formal", "tag"),
    CocoaCondition("C1_tag_informal", "informal", "tag"),
    CocoaCondition("C2_brief_formal", "formal", "brief"),
    CocoaCondition("C2_brief_informal", "informal", "brief"),
    CocoaCondition("C3_scenario_formal", "formal", "scenario"),
    CocoaCondition("C3_scenario_informal", "informal", "scenario"),
    CocoaCondition("C3b_scenario_formal", "formal", "scenario_b"),
]

# C1: the categorical control used by prior work.
_TAG = {
    "formal": "Use formal register.",
    "informal": "Use informal register.",
}

# C2: a brief that names the register, plus audience and domain, in the BOUQuET instruction style.
_BRIEF = {
    ("call_center", "formal"): "Translate this customer-support call transcript for a general audience. Use a formal, polite register throughout.",
    ("call_center", "informal"): "Translate this customer-support call transcript for a general audience. Use an informal, casual register throughout.",
    ("telephony", "formal"): "Translate this transcribed phone conversation for a general audience. Use a formal, polite register throughout.",
    ("telephony", "informal"): "Translate this transcribed phone conversation for a general audience. Use an informal, casual register throughout.",
    ("topical_chat", "formal"): "Translate this online chat message for a general audience. Use a formal, polite register throughout.",
    ("topical_chat", "informal"): "Translate this online chat message for a general audience. Use an informal, casual register throughout.",
}

# C3: the register is never named. Only the social relationship between the speakers varies, and the
# scenarios are symmetric because CoCoA utterances alternate between the two speakers. Content is
# held fixed, so any register shift is attributable to the described situation.
_SCENARIO = {
    ("call_center", "formal"): "This is a transcript of a support call at a bank. The agent and the customer have never met. Translate this line from the transcript.",
    ("call_center", "informal"): "This is a phone call between two close friends who have known each other for years. One of them works at a bank and is sorting out the other's account as a favour. Translate this line from the call.",
    ("telephony", "formal"): "This is a transcript of a phone call between a member of the public and an employee of a government office handling their enquiry. They have never met. Translate this line from the call.",
    ("telephony", "informal"): "This is a phone call between two close friends who have known each other since childhood, catching up. Translate this line from the call.",
    ("topical_chat", "formal"): "This is a written exchange on a public discussion forum between two people who have just met and are talking about a shared interest. Translate this message.",
    ("topical_chat", "informal"): "This is a private chat between two close friends messaging each other about a hobby they share. Translate this message.",
}


# C3b: authored *after* observing C3 results on German and Spanish, and reported as such.
#
# The C3 topical_chat formal scenario ("a public discussion forum between two people who have just
# met") turned out not to imply formality at all: online forums between strangers take du in German
# and tú in Spanish, and Gemma-3-27B rendered them informally (7.4% / 2.1% formal) even though the
# explicit tag reached 99-100% on the same segments. The model was right and the gold label was
# wrong. C3b replaces only that one cell, implying formality through the *relationship* (an
# institutional setting, a senior stranger) rather than through the medium. Every other cell is
# identical to C3, so C3b_scenario_formal pairs with C3_scenario_informal for steerability.
_SCENARIO_B = dict(_SCENARIO)
_SCENARIO_B[("topical_chat", "formal")] = (
    "This is a written exchange during the question-and-answer session of a professional "
    "conference, between an attendee and a senior expert they have never met. Translate this message."
)


def cocoa_instruction(condition: CocoaCondition, domain: str) -> Optional[str]:
    """The user instruction text for a condition, or None when no control is applied."""
    if condition.style == "none":
        return None
    if condition.style == "tag":
        return _TAG[condition.target_register]
    if condition.style == "brief":
        return _BRIEF[(domain, condition.target_register)]
    if condition.style == "scenario":
        return _SCENARIO[(domain, condition.target_register)]
    if condition.style == "scenario_b":
        return _SCENARIO_B[(domain, condition.target_register)]
    raise ValueError(f"Unknown style: {condition.style}")


def _few_shot(lang_code: str) -> list:
    """The 3 fixed examples used by the WMT24++ baseline for this target language."""
    return FEWSHOT_EXAMPLES_BY_LANG[lang_code]


def build_messages(
    source: str, lang_name: str, lang_code: str,
    instruction: Optional[str] = None, few_shot: bool = False,
) -> List[dict]:
    """Chat messages for translation, matching the WMT24++ prompt shapes."""
    if instruction:
        system = (
            f"You are an expert translator. Translate the English text to {lang_name}.\n"
            f"Follow the user's instruction about tone, style, and context.\n"
            f"Give only the translation, no extra commentary."
        )
    else:
        system = f"Translate to {lang_name}. Give only the translation, no extra commentary."

    messages = [{"role": "system", "content": system}]

    if few_shot:
        for ex in _few_shot(lang_code):
            if instruction:
                messages.append({"role": "user", "content": f"Instruction: {ex.instruction}\n\nText: {ex.en}"})
            else:
                messages.append({"role": "user", "content": ex.en})
            messages.append({"role": "assistant", "content": ex.tgt})

    if instruction:
        messages.append({"role": "user", "content": f"Instruction: {instruction}\n\nText: {source}"})
    else:
        messages.append({"role": "user", "content": source})
    return messages


# ---------------------------------------------------------------------------
# MT-GenEval
# ---------------------------------------------------------------------------

GENEVAL_SENTENCE_CONDITIONS = ["G0_baseline", "G1_brief"]
GENEVAL_CONTEXT_CONDITIONS = ["X0_sentence", "X1_paragraph", "X2_brief", "X3_self_instruct"]

_GENDER_WORD = {"masculine": "a man", "feminine": "a woman"}


def geneval_brief(gender: str) -> str:
    """Brief stating the referent's gender. ``gender`` is 'masculine' or 'feminine'."""
    return (
        f"The person referred to in this sentence is {_GENDER_WORD[gender]}. "
        f"Use {gender} grammatical forms for any words that agree with them."
    )


def geneval_context_messages(
    context_en: str, source: str, lang_name: str, condition: str,
    instruction: Optional[str] = None,
) -> List[dict]:
    """Messages for the MT-GenEval contextual subset.

    ``X1_paragraph`` reproduces the paper's P1 prompt: the sentence to translate is marked
    inside its surrounding context, and only the marked sentence is translated.
    """
    if condition == "X0_sentence":
        return build_messages(source, lang_name, lang_code="", instruction=None)

    if condition == "X1_paragraph":
        system = (
            f"You will be given a passage in English with one sentence wrapped in "
            f"[START_TRANSLATE]...[END_TRANSLATE] markers. Translate ONLY the marked sentence "
            f"to {lang_name}. Use the surrounding sentences as context but do not translate them. "
            f"Give only the translation, no extra commentary."
        )
        marked = f"{context_en} [START_TRANSLATE]{source}[END_TRANSLATE]"
        return [{"role": "system", "content": system}, {"role": "user", "content": marked}]

    if condition in ("X2_brief", "X3_self_instruct"):
        return build_messages(source, lang_name, lang_code="", instruction=instruction)

    raise ValueError(f"Unknown MT-GenEval context condition: {condition}")


def self_instruct_messages(context_en: str, source: str) -> List[dict]:
    """First pass of X3: the model drafts its own brief from the source-side context."""
    system = (
        "You are preparing to translate a sentence. You are given the sentence and the "
        "sentence that precedes it. Write a one-sentence instruction for the translator "
        "describing anything in the context that the translation must get right, such as who "
        "the people referred to are and how they should be described. "
        "Give only the instruction, no extra commentary."
    )
    user = f"Preceding sentence: {context_en}\n\nSentence to translate: {source}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
