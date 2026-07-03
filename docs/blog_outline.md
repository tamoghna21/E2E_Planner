# Blog outline — the "better model, worse driving" story

The whole post hangs on one counterintuitive beat: **I improved the model and it drove worse.**
Lead with it, pay it off with the DAgger curve, stay honest about scale. Target 1,200–1,800 words
for Medium; the LinkedIn teaser at the bottom links to it.

---

## Title options

- *I made my self-driving model twice as accurate. It drove worse.*
- *Better labels, worse driving: a closed-loop lesson in imitation learning*
- *Why my driving policy got worse when I fixed it — and how DAgger saved it*
- *0% → 67%: debugging an end-to-end driving policy, one diagnosis at a time*

Pick the one that front-loads the paradox. The second is the cleanest hook without implying bad
outcomes.

## Reading the visuals (add this box near the top of the post)

> **How to read the GIFs:** The camera follows the **ego vehicle** — the car being controlled by
> the learned policy. The ego is always centered in frame and always faces upward. Its path is
> marked by a **teal/green trail**; the leading edge of the trail is where the ego currently is.
> Surrounding vehicles (in random colors — orange, blue, pink, etc.) are not controlled by the
> policy; they replay their real logged trajectories from the nuScenes or Waymo dataset.

## The hook (first 2–3 sentences — this is what LinkedIn shows)

Open cold on the paradox, not on setup. Something like: "I halved my model's prediction error and
it started driving off the road more often. That wasn't a bug — it was the most important result in
the project." Then one line on what the project is. **Do not** open with "I've been learning about
autonomous driving." Open with the anomaly.

*Visual here:* the `bc_vs_dagger_waymo0.gif` side-by-side (teal trail short on the left vs long
on the right, same scene). Caption it: "Same held-out scene. Left: plain BC — the teal trail
(ego path) is short; the policy loses control early. Right: after DAgger, the trail runs the full
route — 95.8% completion."

## Act 1 — Setup (keep it short, ~200 words)

- What "end-to-end" and "closed-loop" mean, in one sentence each, for a semi-technical reader.
- The one idea that makes the project modern: **open-loop accuracy is not driving skill.** The
  field moved to closed-loop benchmarks (nuPlan, CARLA Leaderboard, Bench2Drive) for exactly this
  reason. You're about to show why, live.
- Stack in one line: MetaDrive + ScenarioNet replaying real nuScenes/Waymo logs as closed-loop
  digital twins — real scenes, no perception stack.

## Act 2 — The naive attempt, and the first honest failure (~250 words)

- Behavior cloning: imitate the logged ego. Twist: the logs have **no control channel**, so I had
  to *derive* the steering/throttle labels.
- It failed closed-loop. Instead of hand-waving "needs more data," I ran a **per-dimension loss
  check**: the model had learned steering but essentially never learned throttle.
- Root cause: my throttle labels were a heuristic, not a real physics inversion. **The decisive
  test:** replay my own "ground-truth" labels through the simulator's physics — they couldn't
  drive the route either. So BC was never going to work. Label problem, not model problem.
- Takeaway to state explicitly: *diagnose before you scale.* This is the part that signals
  seniority.

## Act 3 — The twist: I fixed it and it got worse (~300 words — the centerpiece)

- The fix: a **reactive tracking-controller expert** (pure-pursuit + PID) that follows the logged
  path but reacts to the car's real state — so it produces physically consistent labels at *any*
  state, not just along the log.
- Result: open-loop validation loss **halved (0.14 → 0.068)**. Textbook improvement.
- And closed-loop performance **dropped** — more scenarios hit the road boundary earlier.

*Visual here:* the `headline_figure.png` two-panel (loss down, driving not). This is the money
figure — give it room and a plain-language caption.

- Why: fixing throttle let the car actually drive at speed, which **unmasked a lateral-control
  problem** that stalling had been hiding. A better model diverged from the route *further from
  the start*, not less.
- Name it: this is **covariate shift** — the policy only ever saw the expert's clean states in
  training, so the moment its own small errors put it in an unfamiliar state, it had no idea how to
  recover. I confirmed it with two diagnostics: lateral offset climbing toward the road edge over
  tens of steps, and an **action-gap** measurement showing the expert would correct hard exactly
  where the policy didn't.

## Act 4 — The fix that matches the diagnosis: DAgger (~300 words)

DAgger (*Dataset Aggregation*, [Ross, Gordon & Bagnell, AISTATS 2011](https://arxiv.org/abs/1011.0686))
is the canonical imitation-learning solution to covariate shift. The idea is simple: instead of
training only on the expert's trajectory, **let the student policy drive, and have the teacher
label the states the student actually reaches**. The student's mistakes and recoveries end up in
the dataset, so the policy learns what to do in the situations it will actually encounter.

Key clarifications worth stating explicitly:
- DAgger is **still imitation learning — no reward signal, no reinforcement learning.** It only
  changes *which states* get expert labels. This is the in-paradigm answer to covariate shift.
- It was only possible here because the reactive expert is **queryable at any state** — it's a
  controller (pure-pursuit + PID), not a replay of a fixed log. That's why fixing the label
  problem in Act 3, which felt like a step backward, was actually what unlocked the real fix.
- The payoff: success **0% → 17% → 67%**, then a plateau.

*Visual here:* the `dagger_progress.png` iteration curve. Call out the plateau explicitly as
evidence: iterations 3–4 added ~1,750 more labeled states and *didn't* improve — so the binding
constraint was **distribution coverage, not data volume.** That's the quantitative version of the
covariate-shift claim, and it means I didn't even need a separate ablation.

*Visual here:* the `bc_vs_dagger4_nuscenes7.gif` — the teal trail on the right runs to completion
where the left trail is short.

## Act 5 — Honesty (short, ~150 words — this section builds trust, don't skip it)

- **Say n = 6 out loud.** "67%" is 4 of 6 scenarios; these are trends, not significance.
- Trained on nuScenes only; Waymo is out-of-domain and geometry affects completion.
- One scenario (`nuscenes:6`) is a tight intersection outside the tiny training set's coverage and
  does not converge at any iteration — it's in the evaluation, not filtered. You can show its
  (short) teal trail as an honest data point.
- Frame the whole thing as a **methodology demonstrator**, not a benchmark.

## Landing (~120 words)

- Bring it back to the thesis: the number that looked like progress (open-loop loss) was the wrong
  number; the one that mattered (closed-loop behavior) told a different story — and that gap *is*
  the modern autonomous-driving research problem in miniature.
- The meta-point for the reader: **the interesting result was a negative one, diagnosed properly.**
  A model that drives cleanly is a demo; understanding *why* one diverges and fixing it on evidence
  is engineering.
- CTA: link the GitHub repo, invite issues/questions.

---

## Visual placement cheat-sheet

1. Top / hook → `bc_vs_dagger_waymo0.gif` (short teal trail on left, long on right)
2. Above Act 1 → the "how to read the visuals" box (ego = teal trail, always centered/facing up)
3. Act 3 → `headline_figure.png` (the centerpiece)
4. Act 4 → `dagger_progress.png` + `bc_vs_dagger4_nuscenes7.gif`
5. Act 5 → a still from `nuscenes:6` (honest failure — the trail barely starts)

**GIFs are already rendered ego-tracked:** tight camera, heading-up, teal trail drawn. Use them
as-is. `showcase_dagger_success.gif` (Waymo success, single-policy clip) is the best LinkedIn
direct-attach video.

## Tone notes

- Write to a smart peer, not a beginner. Explain covariate shift and DAgger in one plain sentence
  each; don't lecture.
- Every claim near a number should carry its own caveat (n, in/out-of-domain) *inline* — pre-empt
  the skeptical reader instead of getting corrected in the comments.
- Avoid "revolutionary/breakthrough." The story is strong *because* it's measured and honest.

---

## LinkedIn teaser post (short version that links to the blog)

> I halved my self-driving model's prediction error — and it performed worse in the actual
> driving test.
>
> That wasn't a bug. It turned out to be the most useful result in the whole project.
>
> I built a small end-to-end driving policy on real nuScenes & Waymo scenarios — trained and
> evaluated **closed-loop** (the model actually drives, instead of just being scored on one-step
> predictions).
>
> The arc:
> • Naive imitation failed → the labels were physically wrong (caught with a per-dimension loss
>   check, not a guess).
> • Fixing the labels *halved* the loss and the policy performed worse closed-loop — a textbook
>   covariate-shift signature, confirmed with lateral-drift and action-gap diagnostics.
> • DAgger (Ross et al., AISTATS 2011) — still pure imitation learning, no RL — took success
>   from 0% → 67%, then plateaued (which is itself the proof it was a distribution problem, not
>   a data-volume one).
>
> The lesson I keep coming back to: the metric that looked like progress was the wrong metric.
> That gap between "accurate" and "actually drives" is basically the modern autonomous-driving
> research problem in miniature.
>
> Full writeup + code + videos 👉 [link]
>
> #autonomousdriving #machinelearning #imitationlearning #robotics

*(Attach `showcase_dagger_success.gif` directly to the LinkedIn post — the ego's teal trail
navigating through real Waymo traffic is the visual that earns the click. Native video/GIF
outperforms a link preview in reach.)*
