# Mentorra — The AI Founder Boardroom

**Mentorra** is a **prompt-native AI boardroom** that gives founders world-class guidance and a concrete **30-day execution plan**.  
Instead of generic chat advice, Mentorra assembles expert AI mentor agents that collaborate, resolve tradeoffs, and turn strategy into action.

> Prompts are the source of truth. Intelligence is regenerable.

---

## 🚀 Why Mentorra

Founders don’t fail because they lack tools.  
They fail because they lack **judgment, clarity, and prioritization**.

Most AI tools:
- Provide generic advice
- Don’t resolve conflicting guidance
- Aren’t grounded in real-world evidence
- Don’t convert thinking into execution

Mentorra fixes this by simulating a **real founder boardroom** — on demand.

---

## 🧠 What Mentorra Does

Mentorra:
1. Collects a founder’s context (idea, stage, constraints)
2. Selects the right AI mentors using a router agent
3. Runs mentors in parallel for independent judgment
4. Synthesizes everything into a **single, decisive 30-day plan**
5. (Optional) Grounds decisions with live competitor and pricing data
6. (Optional) Delivers advice and plans via **high-quality voice mentors**

---

## 🎙️ Voice Mentors (Optional)

Mentorra supports **voice delivery** of mentor insights and final plans using ElevenLabs.

Why voice?
- Feels like a private call with a trusted mentor
- Conveys urgency, confidence, and nuance
- Enables listening while walking or driving

---

## 🧩 Architecture Overview

Mentorra is built using **Prompt-Driven Development (PDD)**.

### Core Pipeline
Founder Intake
↓
Router Agent
↓
Mentor Agents (parallel)
↓
Synthesis Agent
↓
30-Day Execution Plan

### Key Principles
- Prompts define behavior
- Code is a regenerable artifact
- Tests accumulate and prevent regressions
- Changes happen at the prompt level, not via patching

---

## 🤖 AI Mentor Agents

Mentors are **pattern-based personas**, not impersonations.

Initial set:
- **Adrian Insight** — startup fundamentals, PMF, focus
- **Katerina Catalyst** — first revenue, sales, pricing, resilience
- **Sophia Architect** — experience, trust, narrative, differentiation
- **Vincent Forge** — startup fundamentals, The Impossible Builder

Mentors produce structured briefs with:
1. Diagnosis  
2. Key Insight  
3. What You’re Likely Doing Wrong  
4. What You Should Do Instead  
5. Immediate Action (This Week)

---

## 🌍 Real-World Grounding (Proof Pack)

Mentorra can pull live market data using **Rtrvr.ai**:
- Competitors
- Pricing models
- Positioning language
- Source URLs

This **Proof Pack** is injected into synthesis so plans reflect real market conditions.

---

## 🛠️ Tech Stack

- **Frontend:** Next.js (App Router), React, plain CSS
- **Backend:** Node.js API routes
- **AI Models:** LLM-based (router, mentors, synthesis)
- **Web Retrieval:** Rtrvr.ai
- **Voice:** ElevenLabs
- **Methodology:** Prompt-Driven Development (PDD)

---

## 📁 Repository Structure

/app
/boardroom
page.tsx
/components
FounderIntakeForm.tsx
MentorCards.tsx
SynthesisPlan.tsx
ProofPackButton.tsx
/prompts
router.prompt
mentor_adrian.prompt
mentor_helena.prompt
mentor_sophia.prompt
synthesis.prompt
/lib
proofPackToSynthesis.ts
/tests
router.test.ts
mentor_format.test.ts
synthesis_format.test.ts

---

## 🧪 Testing & Reliability

Mentorra uses **test accumulation** to ensure stability:
- Router JSON schema validation
- Mentor output structure tests
- Synthesis heading and format tests
- Proof Pack injection regression tests

These tests act as **“mold walls”** that prevent bugs from reappearing after regeneration.

---

## 🏆 Hackathon Focus

For the hackathon, we intentionally:
- Avoid auth and payments
- Optimize for clarity and demo impact
- Emphasize PDD rigor and agent orchestration
- Showcase real-world grounding and voice delivery

---

## 👥 Who This Is For

- Early-stage founders
- Builders exploring PDD and agentic systems
- Accelerators and incubators
- Investors interested in AI-native products

---

## 🤝 Contributing

We welcome contributors in:
- Frontend (UI polish)
- Backend (orchestration, reliability)
- Agent logic (prompt tuning, tests)
- Demo & pitch storytelling

Open an issue or jump into the discussion.

---

## 📌 Vision

Mentorra starts with founders — and expands to:
- Executives
- Product leaders
- Creators
- Career transitions
- High-stakes decision-making

> The future won’t be built by people thinking alone.

---

## 📫 Contact
For questions, demos, or collaboration:
- Open an issue
- Reach out via the hackathon Discord

---

**Mentorra**  
*World-class mentorship. Regenerable intelligence. Real execution.*
# mentorrahackaton
