# Short answers

**Q1 — If you only had 200 labeled replies, how would you improve the model without collecting thousands more?**  
Use data-efficient techniques: fine-tune pre-trained transformer carefully, apply data augmentation (paraphrasing/back-translation), use active learning, and ensemble models or cross-validation to stabilize predictions.

**Q2 — How would you ensure your reply classifier doesn’t produce biased or unsafe outputs in production?**  
Remove PII, check dataset for label bias, evaluate per-group performance, log predictions, set low-confidence thresholds for human review, and monitor drift continuously.

**Q3 — Prompt design strategies for generating personalized cold email openers with an LLM?**  
Use structured prompts with placeholders (company, role), include few-shot examples, constrain style/length, explicitly forbid generic phrases, and validate outputs against factual context.
