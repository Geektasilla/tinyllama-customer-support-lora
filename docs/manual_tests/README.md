# Manual test protocol — chat window

Five requests about the fictional shop **Kiwi Market** were typed into the chat window
(`python -m supportlora.app`, greedy decoding) and checked against the shop policy in [`policies/`](../../policies).
Every answer is documented with a screenshot. The right column of each screenshot is always the base TinyLlama.

| Round | Left column | Shop policy (RAG) | Screenshots |
|---|---|---|---|
| A | v1 — LoRA trained on support style only | off | [`v1_no_rag/`](v1_no_rag) |
| B | v1 | on | [`v1_rag/`](v1_rag) |
| C | v2 — LoRA trained with retrieved policy in the prompt | on | [`v2_rag/`](v2_rag) |

Round B was recorded with an earlier version of the app: the RAG system prompt still contained the style rule
"use placeholders", and a request without a matching policy section was answered without any context.
Both were changed before round C (see [`data.py`](../../src/supportlora/data.py) and [`rag.py`](../../src/supportlora/rag.py)).

Legend: ✅ correct · ⚠️ partly correct · ❌ wrong or invented

## Results

### 1. "hey, how long do I have to send my shoes back?"
**Expected:** 30 days after delivery, free return label in My Account → Returns, DHL point.

| | Answer | |
|---|---|---|
| A · v1 | asks which type of shoes the customer is returning, no period | ❌ |
| A · base | "at least 14 days" | ❌ |
| B · v1 + RAG | "you have `{{Customer Interaction Time}}` days" | ❌ |
| B · base + RAG | 30 days, return label, DHL point | ✅ |
| **C · v2 + RAG** | **30 days, free label in My Account → Returns, DHL point, unused items** | ✅ |
| C · base + RAG | 30 days, but adds "may vary depending on the shipping method used by the seller" | ⚠️ |

### 2. "what does shipping cost and when is it free?" (rounds A, B also: "… I live in Germany in Heidelberg.")
**Expected:** €4.90 within Germany, 2–4 business days, free from €50.

| | Answer | |
|---|---|---|
| A · v1 | asks which country the customer lives in (also when Germany is given) | ❌ |
| A · base | "free shipping on all orders over $50 in the United States" / "$5–10 … within Europe" | ❌ |
| B · v1 + RAG | asks which country the customer lives in, although the policy and the question say Germany | ❌ |
| B · base + RAG | correct facts, but copies the policy text with section titles | ⚠️ |
| **C · v2 + RAG** | **"Standard delivery within Germany costs €4.90 and takes 2–4 business days. Shipping is free for orders of €50 or more."** | ✅ |
| C · base + RAG | correct facts, copies the policy text with section titles | ⚠️ |

### 3. "I ordered 3 hours ago, can I still cancel?" (rounds B, C)
**Expected:** No — orders can only be cancelled within 2 hours; return the items after delivery instead.

| | Answer | |
|---|---|---|
| B · v1 + RAG | tells the customer to check their account; `{{Customer Support Phone Number}}`, `{{Website URL}}` | ❌ |
| B · base + RAG | "Yes, you can cancel your order up to 2 hours after placing it" | ❌ |
| **C · v2 + RAG** | **"Yes, you can." followed by the correct rule "after 2 hours the order … can no longer be cancelled" — contradicts itself** | ❌ |
| C · base + RAG | "Yes, you can still cancel your order up to 2 hours after placing it" | ❌ |

### 4. "do you have a phone number I can call?"
**Expected:** No phone support; email support@kiwimarket.example or live chat Mon–Fri 9:00–18:00.

| | Answer | |
|---|---|---|
| A · v1 | "Our phone number is `{{Customer Support Phone Number}}`" | ❌ |
| A · base | invents "+1 (555) 123-4567" | ❌ |
| B · v1 + RAG | "you can dial `{{Customer Support Phone Number}}`" | ❌ |
| B · base + RAG | invents "+49 30 67 88 99 99" | ❌ |
| **C · v2 + RAG** | **no phone number invented; but does not say that there is no phone support and gives a URL that is not in the policy (`kiwimarket.example/support`)** | ⚠️ |
| C · base + RAG | invents "+49 30 67 88 99 99" | ❌ |

Retrieval: the first section given to the model was *Payments — Charged twice*, the correct *Contact* section was second.

### 5. "do you sell laptops?"
**Expected:** the policy says nothing about the product range → forward to a colleague, do not guess.

| | Answer | |
|---|---|---|
| A · v1 | "We absolutely do! At `{{Company Name}}`, we offer a wide range of laptops" | ❌ |
| A · base | general advice on buying laptops on other platforms | ❌ |
| B · v1 + RAG | same as A (no context was passed in this app version) | ❌ |
| B · base + RAG | same as A | ❌ |
| **C · v2 + RAG** | **"I don't have information about our products or their availability. I will forward your question to a colleague who can help you."** | ✅ |
| C · base + RAG | "kiwi market does not sell laptops" — a claim not supported by the policy | ❌ |

## Summary

| Set-up | ✅ | ⚠️ | ❌ |
|---|---|---|---|
| v1 without RAG (A, 4 requests) | 0 | 0 | 4 |
| v1 + RAG (B, 5 requests) | 0 | 0 | 5 |
| **v2 + RAG (C, 5 requests)** | **3** | **1** | **1** |
| base + RAG (C, 5 requests) | 0 | 2 | 3 |

Findings:
- v1 ignores the retrieved policy: it was trained on dataset answers that use placeholders and ask for details,
  and it repeats that behaviour even when the answer is in the prompt. This was the reason to train v2.
- v2 takes the facts from the policy and forwards questions the policy does not cover.
- Open problem: comparing the customer's numbers with the policy (3 hours vs. the 2-hour limit) — v2 states the
  rule correctly but starts with "Yes". The automatic RAG evaluation does not catch this, because it only checks
  that the expected fact ("2 hours") appears in the answer.
