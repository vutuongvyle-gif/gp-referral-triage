import re

from langchain_core.callbacks import BaseCallbackHandler

from referral_triage.privacy.nhi import is_valid_nhi

NHI_RE = re.compile(r"\b[A-HJ-NP-Z]{3}(?:\d{4}|\d{2}[A-HJ-NP-Z]{2})\b")


class PIILeak(RuntimeError):
    pass


class PIIGuard(BaseCallbackHandler):
    raise_error = True            # make LangChain propagate the exception

    def __init__(self):
        self.blocked = 0

    def on_chat_model_start(self, serialized, messages, **kwargs):
        for batch in messages:
            for m in batch:
                text = m.content if isinstance(m.content, str) else str(m.content)
                for hit in NHI_RE.findall(text):
                    if is_valid_nhi(hit):
                        self.blocked += 1
                        raise PIILeak(f"Unmasked NHI in prompt ({hit[:3]}****)")


PII_GUARD = PIIGuard()