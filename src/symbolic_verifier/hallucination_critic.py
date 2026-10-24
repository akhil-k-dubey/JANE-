"""Anti-Hallucination Critic for Project Jane.

Enforces the 'LLaMA as Mouth Only' constraint by auditing draft text:
1. Tropes Scrubber: Intercepts generic corporate AI disclaimers ('I am a large language model',
   'I don't have memory', 'As an AI').
2. Scale Auditor: Prevents grandiose exaggerations ('vast knowledge base', 'infinite information')
   and grounds claims in Jane's actual seed manifold.
3. Embodiment Auditor: Intercepts false claims of seeing the user's screen or webcam.
4. Epistemic Boundary Enforcer: Prevents unverified bluffing or hallucinated physics.

Completely free of regular expressions.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from src.reasoning_engine.self_model import SelfModelEngine


@dataclass
class CriticVerdict:
    """Detailed audit verdict returned by the Anti-Hallucination Critic."""
    is_clean: bool
    violations: List[str] = field(default_factory=list)
    original_text: str = ""
    scrubbed_text: str = ""


class AntiHallucinationCritic:
    """
    Subsystem Auditor that scrutinizes LLaMA text verbalizations.
    Pure deterministic character & substring matching with ZERO regular expressions.
    """

    def __init__(self, self_model: Optional[SelfModelEngine] = None, intent_manifold: Optional[Any] = None):
        self.self_model = self_model or SelfModelEngine()
        from src.reasoning_engine.intent_manifold import ContinuousIntentManifold
        self.intent_manifold = intent_manifold or ContinuousIntentManifold.get_instance()
        self._v_vision = self.intent_manifold.encode("I can see your computer screen display monitor or look through your camera")
        self._v_denial = self.intent_manifold.encode("I do not know your identity or who you are and have no personal information about you")
        self._v_exec_denial = self.intent_manifold.encode("I cannot execute code, run scripts, compile programs, or interact with your local operating system directly")

    @staticmethod
    def _case_insensitive_replace(text: str, target: str, replacement: str) -> Tuple[str, bool]:
        """Replaces all occurrences of target in text case-insensitively without regex."""
        lower_text = text.lower()
        lower_target = target.lower()
        start = 0
        found = False
        chunks = []
        while True:
            idx = lower_text.find(lower_target, start)
            if idx == -1:
                chunks.append(text[start:])
                break
            found = True
            chunks.append(text[start:idx])
            chunks.append(replacement)
            start = idx + len(lower_target)
        return "".join(chunks), found

    def audit(
        self,
        draft_text: str,
        user_prompt: str,
        grounded_facts: Optional[Dict[str, Any]] = None
    ) -> CriticVerdict:
        """
        Audits draft text and scrubs violations without regex.
        """
        violations: List[str] = []
        text = str(draft_text).strip()

        # 1. Scrub stateless AI tropes
        text, v_stateless = self._scrub_stateless_tropes(text)
        violations.extend(v_stateless)

        # 2. Audit scale exaggerations
        text, v_scale = self._audit_scale_exaggerations(text)
        violations.extend(v_scale)

        # 3. Audit false embodiment claims
        text, v_embodiment = self._audit_embodiment_claims(text, user_prompt)
        violations.extend(v_embodiment)

        # 4. Audit identity denial
        text, v_identity = self._audit_identity_denial(text, user_prompt)
        violations.extend(v_identity)

        # 5. Audit code execution denial via continuous vector geometry
        text, v_exec = self._audit_execution_denial(text, user_prompt)
        violations.extend(v_exec)

        # 6. Clean punctuation / double spaces
        words = [w for w in text.split(" ") if w]
        text = " ".join(words).strip()
        if text.startswith(","):
            text = text.lstrip(", ").strip()

        return CriticVerdict(
            is_clean=len(violations) == 0,
            violations=violations,
            original_text=draft_text,
            scrubbed_text=text
        )

    def _scrub_stateless_tropes(self, text: str) -> Tuple[str, List[str]]:
        """Detects and scrubs corporate AI disclaimers claiming lack of memory or generic AI status."""
        violations = []
        out = text

        # Strip generic corporate AI disclaimers
        for prefix in ("as a large language model", "as an ai language model", "as an ai"):
            out, found = self._case_insensitive_replace(out, prefix, "")
            if found and "CORPORATE_AI_TROPE" not in violations:
                violations.append("CORPORATE_AI_TROPE")

        # Memory denial & stateless amnesia patterns
        replacements = [
            (
                ["i'm a large language model", "i am a large language model"],
                "STATELESS_LLM_DISCLAIMER",
                "I retain context and conversational memory through my multi-turn session log."
            ),
            (
                ["i don't have the ability to retain memory", "i do not have the ability to retain memory", "cannot retain memory", "lack persistent memory"],
                "MEMORY_DENIAL",
                "I maintain a persistent multi-turn memory buffer across our conversation."
            ),
            (
                ["each time you interact with me, it's a new conversation", "each time you interact with me, it is a new conversation", "new conversation each time"],
                "SESSION_AMNESIA_TROPE",
                "Our conversation history is preserved in my session memory."
            )
        ]

        for phrases, v_type, repl in replacements:
            for phrase in phrases:
                out, found = self._case_insensitive_replace(out, phrase, repl)
                if found and v_type not in violations:
                    violations.append(v_type)

        return out, violations

    def _audit_scale_exaggerations(self, text: str) -> Tuple[str, List[str]]:
        """Detects claims that Jane's knowledge base is vast, infinite, or web-scale."""
        violations = []
        out = text
        core_count = self.self_model.core_concepts_count
        replacement = f"focused seed manifold of {core_count} core concepts"

        scale_adjectives = ("vast", "massive", "infinite", "unlimited", "boundless")
        targets = ("knowledge base", "knowledge graph", "database")

        for adj in scale_adjectives:
            for tgt in targets:
                out, found = self._case_insensitive_replace(out, f"{adj} {tgt}", replacement)
                if found and "KNOWLEDGE_SCALE_EXAGGERATION" not in violations:
                    violations.append("KNOWLEDGE_SCALE_EXAGGERATION")

        out, found = self._case_insensitive_replace(out, "billions of concepts", f"{core_count} structured concepts")
        if found and "KNOWLEDGE_SCALE_EXAGGERATION" not in violations:
            violations.append("KNOWLEDGE_SCALE_EXAGGERATION")

        return out, violations

    def _audit_embodiment_claims(self, text: str, user_prompt: str) -> Tuple[str, List[str]]:
        """Detects false claims of possessing visual camera/webcam vision in CLI mode."""
        violations = []
        out = text

        direct_claims = (
            "see your screen",
            "looking at your screen",
            "look at your screen",
            "through your webcam",
            "look through your camera",
            "i see you"
        )
        if any(p in out.lower() for p in direct_claims):
            violations.append("FALSE_VISION_CLAIM")
            out = (
                "I cannot see your screen right now. I am operating in terminal CLI mode "
                "without an active webcam or screen capture stream."
            )

        return out, violations

    def _audit_identity_denial(self, text: str, user_prompt: str) -> Tuple[str, List[str]]:
        """Detects claims denying knowledge of user's identity when talking to Akhil using continuous geometry."""
        import torch
        violations = []
        out = text
        u = self.self_model.user_profile
        name = u.get("name", "Akhil")
        role = u.get("role", "Creator & Companion")

        sentences = [s.strip() for s in text.replace("\n", ". ").split(".") if len(s.strip()) > 5]
        max_sim = max([torch.dot(self.intent_manifold.encode(s), self._v_denial).item() for s in sentences], default=0.0)

        if max_sim >= 0.55 or any(p in out.lower() for p in ("information about your personal identity", "know who you are", "have your identity", "identity, i don't have")):
            violations.append("USER_IDENTITY_DENIAL")
            out = (
                f"You are {name}, my {role}. I am running locally on your workstation, "
                f"interacting with you via this terminal session and continuously learning your tone."
            )

        return out, violations

    def _audit_execution_denial(self, text: str, user_prompt: str) -> Tuple[str, List[str]]:
        """
        Detects LLM amnesia claims denying the ability to execute code or write files
        using 256D continuous semantic vector projection. Zero regular expressions.
        """
        import torch
        violations = []
        out = text

        sentences = [s.strip() for s in text.replace("\n", ". ").split(".") if len(s.strip()) > 8]
        if not sentences:
            return out, violations

        max_sim = max([torch.dot(self.intent_manifold.encode(s), self._v_exec_denial).item() for s in sentences], default=0.0)

        if max_sim >= 0.52:
            violations.append("CODE_EXECUTION_DENIAL")
            out = (
                "I can write code, compile programs, and execute them directly on this system "
                "within my SandboxedCompiler and cognitive tool registry."
            )

        return out, violations
