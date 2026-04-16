# Remote Patient Monitoring LLM Assignment

## 1. Assignment Constraints

The assignment asked for an open-source LLM solution for a Remote Patient Monitoring (RPM) chat workflow under the following constraints:

1. Use English as the input language.
2. Use only open-source models, with model size less than or equal to 9B parameters.
3. Mention the exact model version and the reason for choosing it.
4. Complete the work within 5 days.
5. Use synthetic data only for training and evaluation.

The scenario was to build and evaluate an open-source LLM that can support a safe RPM chat workflow with tool or function calling, and then improve the workflow accuracy through training.

The workflow needed to support:

1. Patient onboarding:
   greet the patient and verify identity using first name, last name, and date of birth through a tool.
2. Device setup:
   support pulse oximeter, blood pressure device, scale, and thermometer with tablet pairing.
3. Troubleshooting device issues.
4. Simple education:
   explain how to take vitals correctly with the supported devices.
5. Closing the chat.

The required tasks in the assignment were:

1. Build an open-source LLM chat pipeline.
2. Create a synthetic dataset to evaluate the workflow and report:
   accuracy, TTFT, and total response time for each workflow step.
3. Create synthetic datasets for SFT and optional preference tuning or another method to improve workflow accuracy.
4. Fine-tune the model and report results for:
   baseline, SFT or LoRA-based training, and optionally DPO.

## 2. Problem Framing

This assignment is not just a free-text chatbot task. It is a structured workflow execution problem.

The model must:

- identify the correct workflow state,
- produce a safe assistant message,
- choose the correct tool call when needed,
- fill the correct arguments for the tool,
- avoid unsafe medical advice,
- and behave consistently across onboarding, setup, troubleshooting, education, closing, and escalation scenarios.

Because of this, the task is closer to a combination of:

- instruction following,
- structured generation,
- slot extraction,
- tool invocation,
- and dialogue state control.

That framing strongly influenced the model choice, dataset design, evaluation metrics, and training strategy.

## 3. Overall Approach

I built the project in stages:

1. Define a structured JSON output format for the RPM assistant.
2. Create a prompt-based baseline using an open-source instruct model.
3. Build synthetic datasets for:
   evaluation, supervised fine-tuning, and preference tuning.
4. Fine-tune the model with parameter-efficient methods.
5. Evaluate baseline and fine-tuned versions.
6. Optionally refine behavior further with DPO.

This staged approach made the work practical within the 5-day timeline and allowed direct comparison between baseline and improved versions.

## 4. Output Format and Chat Pipeline

The assistant was designed to return valid JSON only, in a schema like:

```json
{
  "state": "onboarding | device_setup | troubleshooting | education | closing | escalation",
  "assistant_message": "...",
  "tool_call": {
    "name": "...",
    "arguments": {}
  }
}
```

This structure was chosen because it gives several advantages:

- easier evaluation of correctness,
- cleaner tool-calling integration,
- explicit state tracking,
- less ambiguity than free-text answers,
- and better alignment with the RPM workflow requirements.

The pipeline includes:

- prompt construction,
- optional few-shot examples,
- model inference,
- JSON parsing,
- and safety handling for urgent phrases.

For example, urgent phrases such as chest pain or low oxygen can be routed into an escalation response deterministically instead of relying fully on model generation.

## 5. Model Choice

### Chosen Model

The main model used was:

`Qwen/Qwen2.5-3B-Instruct`

### Why this model was chosen

I considered model families such as:

- Mistral
- Qwen
- Llama

I selected `Qwen2.5-3B-Instruct` for the following reasons:

1. It satisfies the assignment constraint of using an open-source model below 9B parameters.
2. The 3B size is practical for experimentation and fine-tuning within limited hardware.
3. It is already instruction-tuned, which helps on tasks like:
   JSON generation, state-based output, and tool-calling style responses.
4. It has a strong balance between capability and efficiency.
5. It is easier to adapt through LoRA and QLoRA than a larger model.

### Why not a larger Llama model

A larger Llama model could potentially perform well, but it is more memory-heavy and less practical under a short timeline, especially if multiple fine-tuning experiments are required.

### Why not Mistral

Mistral was also a reasonable candidate. However, I preferred Qwen because it generally performs well on instruction following and structured responses, which is important in this workflow because the model must produce strict JSON and correct tool arguments.

## 6. Workflow Coverage

The RPM workflow was decomposed into explicit states:

1. onboarding
2. device_setup
3. troubleshooting
4. education
5. closing
6. escalation

This decomposition helped with:

- prompt design,
- data generation,
- evaluation,
- and training.

For each state, I defined the expected assistant behavior:

- what kind of assistant message should be produced,
- which tool should be called,
- what arguments should be provided,
- and what type of state transition is appropriate.

## 7. Synthetic Dataset Design

Because the assignment required synthetic data only, all datasets were generated programmatically or from workflow rules.

The project uses synthetic data for:

1. evaluation
2. supervised fine-tuning
3. DPO preference tuning

### Why synthetic data made sense here

The workflow is highly structured and rule-driven. That means many behaviors can be defined deterministically:

- if the patient gives name and DOB, call `verify_identity`,
- if the patient wants to pair a device, call `pair_device`,
- if the patient says the device is already paired and wants a reading, call `start_measurement`,
- if the patient reports chest pain or low oxygen, call `escalate_to_nurse`.

That makes synthetic data especially suitable because labels can be generated consistently and safely.

## 8. Technique Used to Generate Synthetic Data

The data generation process combined several techniques:

### 8.1 Template-Based Generation

For each workflow step, I wrote user utterance templates such as:

- onboarding requests,
- identity verification statements,
- device setup requests,
- troubleshooting complaints,
- education questions,
- readiness to measure vitals,
- and escalation symptom reports.

Examples:

- "My name is Emily Davis. My DOB is 12/01/1959."
- "Help me set up my pulse oximeter."
- "The tablet can't find my BP device."
- "How do I take my blood pressure correctly?"
- "I'm having chest pain right now."

This approach gives:

- high control,
- fast iteration,
- easy debugging,
- and reproducibility.

### 8.2 Rule-Based Labeling

Each user template was mapped to an expected workflow output using explicit logic.

That output includes:

- the correct state,
- the correct assistant message,
- the correct tool name,
- and the correct tool arguments.

Because the workflow is deterministic in many places, rule-based labeling is a strong fit.

### 8.3 Slot Filling / Parameterized Generation

Templates were populated with variable values such as:

- first names,
- last names,
- dates of birth,
- device ids,
- measurement types,
- escalation reasons.

This is important because it teaches the model not just static phrasing, but extraction and grounding:

- extract DOB correctly,
- preserve device identifiers,
- choose the correct measurement type,
- and pass the correct arguments to tools.

### 8.4 State-Specific Output Construction

The generated dataset was aligned with state-specific logic:

- onboarding data teaches identity verification
- device setup data teaches pairing
- troubleshooting data teaches whether to retry or escalate support
- education data teaches general device guidance or measurement initiation
- closing data teaches safe conversation ending
- escalation data teaches urgent routing without medical advice

### 8.5 Negative Sample Synthesis for DPO

For DPO, I did not generate random bad outputs. Instead, I created plausible but worse outputs by corrupting correct outputs.

Examples include:

- wrong DOB,
- generic device id instead of extracted device id,
- wrong troubleshooting step,
- wrong measurement type,
- weaker or less specific escalation wording,
- wrong state transition in some cases.

This is better than random negative sampling because DPO works best when the rejected answer is believable but inferior.

## 9. Datasets Created

### 9.1 Evaluation Dataset

The evaluation dataset was built to test the workflow behavior across the required RPM stages.

The purpose of the evaluation dataset is to measure:

- whether the model enters the correct state,
- whether it chooses the correct tool,
- whether tool arguments are correct,
- and whether the assistant message is appropriate.

### 9.2 SFT Dataset

The supervised fine-tuning dataset contains:

- synthetic user messages,
- and the correct target JSON outputs.

This dataset is used to teach the model the intended workflow behavior directly.

### 9.3 DPO Dataset

The DPO dataset contains:

- prompt
- chosen output
- rejected output

The chosen output is the correct structured answer.
The rejected output is a degraded but plausible alternative.

This allows preference learning without needing a separate reward model.

## 10. Fine-Tuning Strategies Considered

Several fine-tuning and adaptation strategies are relevant to this type of project.

### 10.1 Prompt-Only / In-Context Learning

This is the baseline approach.

The model is not trained. Instead, behavior is guided by:

- system instructions,
- schema constraints,
- and few-shot examples.

Advantages:

- no training cost,
- quick to test,
- useful baseline.

Limitations:

- less reliable for strict workflow control,
- can drift in output formatting,
- weaker consistency on tool arguments.

### 10.2 Supervised Fine-Tuning (SFT)

SFT trains the model directly on input-output pairs.

In this project:

- input = patient/user message
- output = correct JSON workflow response

Advantages:

- direct supervision,
- easy to implement,
- strong fit when the correct answer is known.

Limitation:

- it teaches what the correct answer is, but not always how to distinguish among several plausible but slightly different answers.

### 10.3 Full Fine-Tuning

Full fine-tuning updates all model parameters.

Advantages:

- highest adaptation capacity,
- can deeply specialize the model.

Limitations:

- most memory expensive,
- slower,
- more likely to cause CUDA out-of-memory errors,
- less practical within the assignment timeline.

### 10.4 LoRA

LoRA is a parameter-efficient fine-tuning method.

Instead of updating the full model, LoRA inserts trainable low-rank matrices into selected modules while keeping the original weights frozen.

Advantages:

- much lower memory footprint,
- fewer trainable parameters,
- faster experimentation,
- strong practical performance for instruction-following tasks.

This was one of the most suitable strategies for this assignment.

### 10.5 QLoRA

QLoRA combines LoRA with model quantization, typically 4-bit.

Advantages:

- lower VRAM usage,
- ability to fine-tune on hardware where full precision would not fit,
- practical for open-source models in constrained settings.

Tradeoff:

- more setup complexity,
- possible slight quality tradeoffs depending on precision and configuration.

### 10.6 DPO

DPO is Direct Preference Optimization.

Instead of training only on the correct answer, it trains on preference pairs:

- prompt
- chosen answer
- rejected answer

Advantages:

- improves behavior on subtle distinctions,
- helps the model prefer one plausible action over another,
- can be used after SFT as a refinement step.

### 10.7 RLHF

RLHF normally involves:

- collecting preference labels,
- training a reward model,
- then optimizing the policy.

It was not used here because:

- the task is structured,
- good and bad outputs can be defined directly,
- and DPO gives a simpler preference-learning path without an extra reward model stage.

### 10.8 DoRA

DoRA is another parameter-efficient adaptation strategy related to low-rank tuning.

It could be explored in future work, but was not prioritized because LoRA and QLoRA were more established and more practical under the assignment deadline.

### 10.9 Transfer Learning

At a broader level, this entire project is a form of transfer learning:

- start with a pretrained instruct model,
- adapt it to a narrow RPM workflow domain.

## 11. Why I Chose This Fine-Tuning Strategy

The most practical strategy for this assignment was:

1. baseline prompt-only evaluation
2. SFT with LoRA or QLoRA
3. optional DPO refinement

This strategy was chosen because:

- it aligns directly with the assignment requirements,
- it is feasible within 5 days,
- it handles hardware limits more gracefully than full fine-tuning,
- and it matches the structure of the task.

Why not full fine-tuning as the main path:

- too resource heavy,
- less efficient for quick iteration,
- not necessary to demonstrate meaningful improvement.

Why DPO as an optional third stage:

- it is useful for correcting near-miss behaviors,
- especially where two outputs are both plausible but one is clearly better.

## 12. Hyperparameters That Could Be Tuned

Hyperparameter tuning is important for both training quality and feasibility.

### 12.1 General Training Hyperparameters

Important parameters include:

- learning rate
- batch size
- gradient accumulation steps
- number of epochs
- warmup steps or warmup ratio
- weight decay
- optimizer
- scheduler

Why they matter:

- learning rate controls update size,
- batch size affects memory and gradient stability,
- gradient accumulation helps emulate larger effective batch sizes,
- epochs control training duration,
- weight decay helps regularization.

### 12.2 Sequence and Formatting Hyperparameters

- max sequence length
- prompt format
- truncation strategy
- padding side

These matter a lot here because the prompt includes:

- system rules,
- few-shot examples,
- and a structured JSON response.

Longer sequences consume more memory, so sequence length is directly tied to feasibility.

### 12.3 LoRA Hyperparameters

- rank `r`
- alpha
- dropout
- target modules

Typical target modules include:

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`

These determine:

- how much adaptation capacity the LoRA layers have,
- how stable the training is,
- and how many additional trainable parameters are introduced.

### 12.4 QLoRA / Quantization Hyperparameters

- quantization precision
- compute dtype
- use of double quantization
- quantization type

These affect:

- memory efficiency,
- training speed,
- numerical stability.

### 12.5 DPO Hyperparameters

- beta
- batch size
- number of preference examples
- quality and diversity of chosen/rejected pairs

Beta is especially important because it controls how strongly the model is pushed toward chosen over rejected responses.

### 12.6 Inference Hyperparameters

For evaluation and demos, important inference settings include:

- temperature
- top-p
- top-k
- max new tokens

For structured JSON tasks, lower temperature and more deterministic decoding are generally preferred.

## 13. What I Would Tune First

If time is limited, the most impactful hyperparameters to tune first are:

1. learning rate
2. max sequence length
3. batch size
4. gradient accumulation steps
5. LoRA rank
6. number of epochs
7. DPO beta

Reason:

- these strongly affect memory usage,
- convergence,
- stability,
- and final workflow accuracy.

If memory becomes a bottleneck, the first adjustments I would make are:

- reduce max sequence length,
- reduce batch size,
- increase gradient accumulation,
- or use a more memory-efficient fine-tuning path like QLoRA.

## 14. Experiments Run

The assignment asked for reporting results on:

1. baseline
2. SFT or LoRA-based training
3. DPO, optional but preferred

That naturally led to the following experiment design:

### 14.1 Baseline

Use the pretrained instruct model without fine-tuning.

Purpose:

- establish starting performance,
- identify typical failure modes,
- measure raw workflow following ability.

### 14.2 SFT / LoRA Fine-Tuning

Train using synthetic supervised examples.

Purpose:

- teach correct workflow behavior,
- improve structured response reliability,
- improve tool calling and argument quality.

### 14.3 DPO Fine-Tuning

Use synthetic preference pairs.

Purpose:

- refine the model’s choices between near-correct and wrong alternatives,
- reduce subtle workflow errors,
- improve preference for safer and more appropriate structured responses.

## 15. Why These Experiments Were Chosen

These experiments were chosen because they map directly to the assignment goals.

- Baseline tells us how good the model already is.
- SFT tells us how much direct supervision helps.
- DPO tells us whether preference tuning can push behavior further.

This creates a clear progression:

- no task-specific training,
- direct task-specific training,
- then preference refinement.

## 16. Resource and Practical Constraints

A major practical issue in LLM fine-tuning is hardware capacity.

Potential problems include:

- CUDA out-of-memory during full fine-tuning,
- long iteration time,
- limited number of experiment runs within the deadline.

That is why parameter-efficient methods such as LoRA and QLoRA are especially valuable.

If full SFT could not fit, alternatives include:

- reducing sequence length,
- reducing batch size,
- using quantization,
- lowering adapter rank,
- or using a smaller base model.

## 17. Evaluation Metrics Used

The assignment specifically asked for:

- accuracy
- TTFT
- total response time

### 17.1 Accuracy

Accuracy is the most important metric because this is fundamentally a structured workflow correctness problem.

The model must:

- enter the correct state,
- choose the correct tool,
- produce correct arguments,
- and generate an appropriate assistant message.

### 17.2 TTFT

TTFT stands for time to first token.

This measures responsiveness, which matters in a real chat workflow because users expect the system to start responding quickly.

### 17.3 Total Response Time

This measures end-to-end latency.

It matters because even if a model is accurate, it is less practical if the response takes too long.

## 18. Why I Chose These Metrics

These metrics are aligned with the actual problem.

- Accuracy reflects workflow correctness.
- TTFT reflects perceived responsiveness.
- Total response time reflects usability.

They are more meaningful for this assignment than generic language-modeling metrics.

## 19. Why Not BLEU, F1, Exact Match, or Perplexity as Main Metrics

### BLEU

BLEU focuses on n-gram overlap and is more suitable for text similarity tasks such as machine translation.

It is not ideal here because:

- multiple assistant phrasings may be acceptable,
- workflow correctness matters more than wording overlap,
- and BLEU does not capture tool-call correctness well.

### F1

F1 is useful for classification or token-level extraction tasks.

It could be useful for subtasks such as:

- state classification,
- tool-call prediction,
- or argument extraction.

But it is not the best single end-to-end metric for the whole workflow.

### Exact Match

Exact match can be too strict if wording differs while behavior is still correct.

It can still be useful for:

- strict JSON comparison,
- tool argument match,
- or state match.

But as a sole main metric, it is often too rigid.

### Perplexity

Perplexity measures language modeling quality, not task execution quality.

A model can have good perplexity but still call the wrong tool or use the wrong arguments.

So perplexity is less aligned with the assignment goal.

## 20. Additional Metrics That Could Be Added

If expanding this work, more task-specific metrics could be useful:

- state accuracy,
- tool-call accuracy,
- tool-argument accuracy,
- JSON validity rate,
- safety escalation recall,
- exact match on critical slots like DOB and device id.

These would give a more detailed view of model behavior.

## 21. What is DPO

DPO stands for Direct Preference Optimization.

The simplest way to explain it is:

- SFT says: "this is the correct answer"
- DPO says: "between these two answers, this one is better"

So DPO is a preference-based training method.

It teaches the model to assign higher probability to the preferred answer than to a less preferred answer for the same prompt.

## 22. DPO in This Project

In this project, DPO acts as a preference-based correction step for JSON workflow responses.

For a given prompt:

- `chosen` is the correct workflow output,
- `rejected` is a plausible but worse output.

Examples:

- correct DOB vs wrong DOB,
- correct device id vs generic unknown device id,
- correct troubleshooting step vs wrong next step,
- correct measurement type vs swapped measurement type.

This helps the model learn subtle distinctions that are not always captured perfectly by plain SFT.

## 23. Why I Did Not Use a Reward Model

A separate reward model was not used because the workflow is structured enough that preferred and non-preferred outputs can be defined directly.

Reasons:

1. The task has a known output schema.
2. Good and bad outputs are easy to specify.
3. Preference pairs can be generated synthetically.
4. DPO can train directly from chosen/rejected pairs.
5. Adding a reward model would increase complexity without a clear need in this assignment.

A reward model is more useful when:

- outputs are open-ended,
- ranking requires softer human judgments,
- or there are many candidate outputs to compare.

That was not necessary here.

## 24. Strengths of the Approach

This project design has several strengths:

- fully aligned with assignment constraints,
- open-source model under 9B,
- fully synthetic training and evaluation,
- structured and measurable outputs,
- practical under hardware limits,
- safe workflow-oriented design,
- and a clear baseline-to-finetuned comparison path.

## 25. Limitations

It is also important to mention limitations honestly:

- synthetic data may not reflect full real-world language diversity,
- the model may overfit template styles if variation is limited,
- preference pairs are heuristic rather than human-ranked,
- evaluation may not cover every edge case,
- full fine-tuning was not explored as deeply due to hardware constraints.

## 26. Future Improvements

If more time were available, future improvements could include:

- more diverse paraphrases,
- typo and noisy-input augmentation,
- more ambiguous user cases,
- deeper comparison of LoRA vs QLoRA vs full fine-tuning,
- richer DPO preference pairs,
- more task-specific metrics,
- and broader error analysis by workflow state.

## 27. Final Conclusion

This assignment was successfully framed as a structured workflow modeling problem rather than an open-ended chat generation problem.

The key design choices were:

- use a compact open-source instruct model,
- define a strict JSON schema,
- generate synthetic workflow-aligned datasets,
- fine-tune with practical parameter-efficient methods,
- evaluate with workflow-relevant metrics,
- and optionally refine with DPO.

Overall, this approach demonstrates that a compact open-source LLM can be adapted for a Remote Patient Monitoring workflow with:

- safe behavior,
- structured tool calling,
- measurable workflow accuracy,
- and practical training under real compute constraints.
