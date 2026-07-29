# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "altair>=5.5,<6",
#   "marimo>=0.14,<1",
#   "pandas>=2.2,<3",
# ]
# ///

import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import altair as alt
    import marimo as mo
    import pandas as pd

    return alt, mo, pd


@app.cell
def _(mo):
    mo.md(r"""
    # Relevance as an execution prior

    Search agents can inspect a corpus with tools such as `ripgrep`, but arbitrary
    traversal may hide useful evidence deep in the collection. The paper
    *A New Role for Relevance* proposes using relevance to guide the interaction:
    order documents, seed a promising paragraph, and rerank local matches.

    **Verdict — partially reproduced.** On 32 fixed BrowseComp-Plus questions in
    the released 100K corpus, the relevance-guided variants used fewer search
    steps and achieved higher open-judge accuracy. The evidence-arrival mechanism
    was strong; the answer gain was small, heterogeneous across slices, and used
    Qwen3 substitutes rather than the paper's proprietary models.
    """)
    return


@app.cell
def _(pd):
    conditions = pd.DataFrame(
        [
            {"condition": "DCI", "judge_accuracy": 3.125, "exact_accuracy": 3.125, "steps": 3.21875, "matches": 88.5, "evidence": 3.125},
            {"condition": "RARG", "judge_accuracy": 6.25, "exact_accuracy": 3.125, "steps": 1.84375, "matches": 51.03125, "evidence": 15.625},
            {"condition": "RARG+", "judge_accuracy": 12.5, "exact_accuracy": 3.125, "steps": 1.5, "matches": 46.0, "evidence": 18.75},
            {"condition": "RARG++", "judge_accuracy": 15.625, "exact_accuracy": 6.25, "steps": 1.53125, "matches": 44.96875, "evidence": 18.75},
        ]
    )
    palette = ["#8B95A5", "#2563EB", "#7C3AED", "#E11D48"]
    order = ["DCI", "RARG", "RARG+", "RARG++"]
    return conditions, order, palette


@app.cell
def _(alt, conditions, order, palette):
    frontier = (
        alt.Chart(conditions)
        .mark_line(point=alt.OverlayMarkDef(size=150), strokeDash=[6, 6], strokeWidth=2)
        .encode(
            x=alt.X("steps:Q", title="Mean tool steps (lower is better)", scale=alt.Scale(domain=[1.3, 3.4])),
            y=alt.Y("judge_accuracy:Q", title="Open-judge accuracy (%)", scale=alt.Scale(domain=[0, 18])),
            color=alt.Color("condition:N", sort=order, scale=alt.Scale(domain=order, range=palette), legend=None),
            order=alt.Order("steps:Q", sort="descending"),
            tooltip=["condition", alt.Tooltip("judge_accuracy:Q", format=".1f"), alt.Tooltip("exact_accuracy:Q", format=".1f"), alt.Tooltip("steps:Q", format=".2f")],
        )
        .properties(width=650, height=360, title="Observed accuracy–efficiency frontier (n=32)")
    )
    labels = (
        alt.Chart(conditions)
        .mark_text(dx=12, dy=-12, fontWeight="bold")
        .encode(x="steps:Q", y="judge_accuracy:Q", text="condition:N", color=alt.Color("condition:N", scale=alt.Scale(domain=order, range=palette), legend=None))
    )
    frontier + labels
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## How the test was matched

    All four conditions used the same first 32 rows of
    `data/bcplus_qa_sample100.jsonl`, all 100,195 exported documents, a six-step
    agent budget, and the same models. DCI traversed the corpus without relevance;
    RARG ordered documents with Qwen3-Embedding-0.6B; RARG+ added eight entry
    paragraphs; RARG++ reranked a bounded 120-match pool. Qwen3-8B acted and
    supplied the open judge, while exact match and known-gold evidence recovery
    served as controls.

    Formal runs used **Kubernetes**, four NVIDIA RTX PRO 6000 Blackwell GPUs per
    condition, and **16 GPUs at peak concurrency**. The queue runner observed
    **1.225558 hours of Kubernetes campaign wall time**, including setup and
    evidence-producing runs.
    """)
    return


@app.cell
def _(pd):
    slices = pd.DataFrame(
        [
            {"slice": "Rows 1–16", "condition": "DCI", "accuracy": 0.0},
            {"slice": "Rows 1–16", "condition": "RARG", "accuracy": 0.0},
            {"slice": "Rows 1–16", "condition": "RARG+", "accuracy": 0.0},
            {"slice": "Rows 1–16", "condition": "RARG++", "accuracy": 0.0},
            {"slice": "Rows 17–32", "condition": "DCI", "accuracy": 6.25},
            {"slice": "Rows 17–32", "condition": "RARG", "accuracy": 12.5},
            {"slice": "Rows 17–32", "condition": "RARG+", "accuracy": 25.0},
            {"slice": "Rows 17–32", "condition": "RARG++", "accuracy": 31.25},
        ]
    )
    return (slices,)


@app.cell
def _(alt, order, palette, slices):
    (
        alt.Chart(slices)
        .mark_bar()
        .encode(
            x=alt.X("condition:N", sort=order, title=None),
            y=alt.Y("accuracy:Q", title="Open-judge accuracy (%)", scale=alt.Scale(domain=[0, 35])),
            color=alt.Color("condition:N", sort=order, scale=alt.Scale(domain=order, range=palette), legend=None),
            column=alt.Column("slice:N", title=None),
            tooltip=["slice", "condition", alt.Tooltip("accuracy:Q", format=".1f")],
        )
        .properties(width=260, height=280, title="The answer gain was concentrated in the second fixed slice")
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    The slice comparison is an important restraint on the headline: all four
    methods scored 0% in the first slice. RARG+ and RARG++ nevertheless recovered
    gold evidence for 12.5% there, showing that retrieving evidence did not ensure
    that this smaller agent synthesized the answer.

    ## Does relevance actually move evidence forward?

    A separate paired diagnostic ranked every known gold document without asking
    the agent to answer. Pooled across 32 questions, the median gold rank moved
    from **25,912.5 under lexicographic traversal to 154.5 under relevance
    traversal**, a 122× median speedup. Gold-document recall reached 37.5% at
    rank 100, 65.6% at 1,000, and 93.8% at 10,000.
    """)
    return


@app.cell
def _(alt, pd):
    visibility = pd.DataFrame(
        [
            {"slice": "Rows 1–16", "view": "Paragraph entry", "visible": 0.0},
            {"slice": "Rows 1–16", "view": "Ordered top 30", "visible": 6.25},
            {"slice": "Rows 1–16", "view": "Reranked top 30", "visible": 12.5},
            {"slice": "Rows 17–32", "view": "Paragraph entry", "visible": 6.25},
            {"slice": "Rows 17–32", "view": "Ordered top 30", "visible": 25.0},
            {"slice": "Rows 17–32", "view": "Reranked top 30", "visible": 31.25},
            {"slice": "Pooled", "view": "Paragraph entry", "visible": 3.125},
            {"slice": "Pooled", "view": "Ordered top 30", "visible": 15.625},
            {"slice": "Pooled", "view": "Reranked top 30", "visible": 21.875},
        ]
    )
    (
        alt.Chart(visibility)
        .mark_bar()
        .encode(
            x=alt.X("slice:N", sort=["Rows 1–16", "Rows 17–32", "Pooled"], title=None),
            xOffset=alt.XOffset("view:N", sort=["Paragraph entry", "Ordered top 30", "Reranked top 30"]),
            y=alt.Y("visible:Q", title="Questions with gold visible (%)", scale=alt.Scale(domain=[0, 35])),
            color=alt.Color(
                "view:N",
                sort=["Paragraph entry", "Ordered top 30", "Reranked top 30"],
                scale=alt.Scale(range=["#D97706", "#0F766E", "#E11D48"]),
                title=None,
            ),
            tooltip=["slice", "view", alt.Tooltip("visible:Q", format=".1f")],
        )
        .properties(width=650, height=320, title="Local reranking raised early gold-evidence visibility")
    )
    return


@app.cell
def _(conditions, mo):
    dci_matches = float(conditions.loc[conditions["condition"] == "DCI", "matches"].iloc[0])
    rargpp_matches = float(conditions.loc[conditions["condition"] == "RARG++", "matches"].iloc[0])
    reduction = 100 * (1 - rargpp_matches / dci_matches)
    mo.md(
        f"""
        ## Interpretation and limits

        The first target claim is **aligned under this setup**: document ordering moved
        known evidence forward, raised evidence recovery from 3.1% to 15.6%, reduced
        mean tool steps from 3.22 to 1.84, and improved judged accuracy.

        The second claim is **partially aligned**: paragraph seeding and narrow local
        reranking improved pooled judged accuracy beyond RARG, and reranking raised
        first-30 gold visibility from 15.6% to 21.9%. Yet entry visibility itself was
        only 3.1%, evidence recall was unchanged from RARG+ to RARG++, and a separate
        500-match pool erased the first-slice gain. The bounded RARG++ condition
        exposed **{reduction:.1f}% fewer matches than DCI**.

        Absolute accuracy should not be compared directly with the paper's 78–84%:
        it used 100 questions and GPT-5.4-mini-family components, while this bounded
        reproduction used 32 questions and Qwen3-8B for acting and judging. There
        were no repeated stochastic seeds, BRIGHT was omitted, and the 1M corpus was
        not attempted. The strongest conclusion is therefore about evidence arrival
        and interaction cost; the answer benefit remains provisional.
        """
    )
    return


if __name__ == "__main__":
    app.run()
