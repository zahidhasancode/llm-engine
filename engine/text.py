"""Turning a stream of tokens into text without breaking characters.

One character can span several tokens. An emoji is four bytes in UTF-8, and a tokenizer may
give each byte its own token. Decoding token by token would show broken characters, so text
is held back until it decodes cleanly.
"""

REPLACEMENT = "�"  # what decoding gives for incomplete UTF-8


class IncrementalDecoder:
    """Decode tokens one at a time, emitting only complete characters.

    Some tokenizers change spacing depending on neighbouring tokens, so each new token is
    decoded together with the tokens just before it, and only the new text is emitted.
    """

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.tokens: list[int] = []
        self.prefix = 0  # where the context window starts
        self.emitted = 0  # how many tokens have become text

    def _decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def push(self, token: int) -> str:
        """Add a token, and return whatever new text is now complete."""
        self.tokens.append(token)
        before = self._decode(self.tokens[self.prefix : self.emitted])
        now = self._decode(self.tokens[self.prefix :])
        if now.endswith(REPLACEMENT) or len(now) <= len(before):
            return ""
        self.prefix, self.emitted = self.emitted, len(self.tokens)
        return now[len(before) :]

    def flush(self) -> str:
        """End of the stream: return anything held back, even if incomplete."""
        before = self._decode(self.tokens[self.prefix : self.emitted])
        now = self._decode(self.tokens[self.prefix :])
        self.prefix = self.emitted = len(self.tokens)
        return now[len(before) :]
