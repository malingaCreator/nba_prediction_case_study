"""
Next-Best-Action Engine — demo UI.

Run with:  streamlit run app.py

Reads the artifacts produced by the notebook:
    predictions.parquet      scored candidates (ncodpers, month, product, score, added)
    santander_sample.parquet customer attributes

Consent and contact history are SYNTHETIC, regenerated here with the same
seed used in the notebook so the app and the analysis agree. See the
framing doc, s4.1 — no synthetic field is presented as observed data.
"""

import numpy as np
import pandas as pd
import streamlit as st

from arbitration import (
    ACTION_VALUE, CONTACT_COST, CHANNEL_PLAN, MIN_PROB, MAX_CONTACTS,
    Candidate, CustomerContext, arbitrate, consent_compliance_check,
)

SEED = 20260914

LABELS = {
    "ind_recibo_ult1":   "Direct debit set-up",
    "ind_cco_fin_ult1":  "Current account",
    "ind_nomina_ult1":   "Payroll deposit",
    "ind_nom_pens_ult1": "Pension deposit",
    "ind_ecue_fin_ult1": "e-Account",
}
CHANNEL_LABELS = {
    "inapp": "In-app inbox", "push": "Push notification",
    "email": "Email", "outreach": "Contact centre / branch",
}

st.set_page_config(page_title="Next-Best-Action Engine", layout="wide")


# ----------------------------------------------------------------- data load

@st.cache_data(show_spinner="Loading scored candidates…")
def load():
    P = pd.read_parquet("predictions.parquet")
    last_month = sorted(P["month"].unique())[-1]
    P = P[P["month"] == last_month].copy()

    df = pd.read_parquet("santander_sample.parquet")
    df["month"] = pd.to_datetime(df["fecha_dato"]).dt.to_period("M")
    attrs = (df[df["month"] == last_month]
             .drop_duplicates("ncodpers")
             .set_index("ncodpers")[["age", "sexo", "renta", "antiguedad",
                                     "segmento", "nomprov",
                                     "ind_actividad_cliente"]])

    rng = np.random.default_rng(SEED)
    ids = P["ncodpers"].drop_duplicates().sort_values().reset_index(drop=True)
    n = len(ids)
    consent = pd.DataFrame({
        "ncodpers": ids,
        "email":    rng.random(n) < 0.72,
        "push":     rng.random(n) < 0.55,
        "inapp":    rng.random(n) < 0.88,
        "outreach": rng.random(n) < 0.31,
        "contacts_this_period": rng.poisson(1.3, n),
    }).set_index("ncodpers")

    held = {p: set(P.loc[P["product"] == p, "ncodpers"]) for p in ACTION_VALUE}
    return P, attrs, consent, held, str(last_month)


def context_for(cid, consent):
    r = consent.loc[cid]
    return CustomerContext(
        customer_id=int(cid),
        consent={k: bool(r[k]) for k in ("email", "push", "inapp", "outreach")},
        contacts_this_period=int(r["contacts_this_period"]),
    )


def candidates_for(cid, P):
    g = P[P["ncodpers"] == cid]
    return [Candidate(product=r.product, score=float(r.score)) for r in g.itertuples()]


@st.cache_data(show_spinner="Running arbitration over all customers…")
def run_all(min_prob, max_contacts):
    P, attrs, consent, _, _ = load()
    by_cust = {cid: g for cid, g in P.groupby("ncodpers")}
    decisions, contexts = [], []
    for cid, g in by_cust.items():
        ctx = context_for(cid, consent)
        cands = [Candidate(product=r.product, score=float(r.score)) for r in g.itertuples()]
        contexts.append(ctx)
        decisions.append(arbitrate(ctx, cands, min_prob=min_prob, max_contacts=max_contacts))
    rows = [{"ncodpers": d.customer_id, "contacted": not d.suppressed,
             "action": d.action, "channel": d.channel,
             "ev": d.expected_value, "p": d.probability, "reason": d.reason}
            for d in decisions]
    return pd.DataFrame(rows).set_index("ncodpers"), consent_compliance_check(decisions, contexts)


P, ATTRS, CONSENT, HELD, LAST_MONTH = load()


# -------------------------------------------------------------------- layout

st.title("Next-Best-Action Engine")
st.caption(
    f"Retail banking · decision month {LAST_MONTH} · "
    "propensity models and evaluation use observed Santander data; "
    "consent, contact history and action values are synthetic (see framing doc)."
)

with st.sidebar:
    st.header("Policy levers")
    min_prob = st.slider("Minimum acceptance probability", 0.0, 0.05,
                         float(MIN_PROB), 0.002, format="%.3f",
                         help="Below this, a contact is not worth the attention cost.")
    max_contacts = st.slider("Contact cap per period", 0, 6, int(MAX_CONTACTS))
    st.divider()
    st.caption("Assumed action values (not measured)")
    st.dataframe(pd.Series({LABELS[k]: f"${v:,.0f}" for k, v in ACTION_VALUE.items()},
                           name="value"), use_container_width=True)

customer_tab, ops_tab = st.tabs(["Customer view", "Operations view"])


# --------------------------------------------------------------- customer view

with customer_tab:
    ids = P["ncodpers"].drop_duplicates().sort_values().tolist()

    c1, c2 = st.columns([2, 1])
    with c1:
        cid = st.selectbox("Customer", ids, index=0,
                           format_func=lambda x: f"#{x}")
    with c2:
        if st.button("Random customer", use_container_width=True):
            st.session_state["cid"] = int(np.random.choice(ids))
            st.rerun()
    cid = st.session_state.get("cid", cid)

    ctx = context_for(cid, CONSENT)
    decision = arbitrate(ctx, candidates_for(cid, P),
                         min_prob=min_prob, max_contacts=max_contacts)

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Customer")
        a = ATTRS.loc[cid] if cid in ATTRS.index else None
        if a is not None:
            st.write(
                pd.Series({
                    "Age": int(a["age"]) if pd.notna(a["age"]) else "—",
                    "Tenure (months)": int(a["antiguedad"]) if pd.notna(a["antiguedad"]) else "—",
                    "Segment": a["segmento"] or "—",
                    "Region": a["nomprov"] or "—",
                    "Active": "Yes" if a["ind_actividad_cliente"] == 1 else "No",
                }, name="")
            )
        st.caption("Consent (synthetic)")
        st.write(pd.Series({CHANNEL_LABELS[k]: ("Yes" if ctx.consent[k] else "No")
                            for k in ctx.consent}, name=""))
        st.metric("Contacts used this period",
                  f"{ctx.contacts_this_period} / {max_contacts}")

    with right:
        st.subheader("Decision")
        if decision.suppressed:
            st.warning("**No action**")
            st.write(decision.reason)
            st.caption(
                "Suppression is a first-class outcome. A contact costs attention, "
                "and the channel erodes if it is spent on weak recommendations."
            )
        else:
            st.success(f"**{LABELS.get(decision.action, decision.action)}**")
            m1, m2, m3 = st.columns(3)
            m1.metric("Channel", CHANNEL_LABELS[decision.channel])
            m2.metric("P(accept)", f"{decision.probability:.2%}")
            m3.metric("Expected value", f"${decision.expected_value:,.2f}")
            st.write(decision.reason)
            if decision.needs_followup:
                st.info("Flagged for contact-centre or branch follow-up (high expected value).")

    st.divider()
    st.subheader("Why — decision trace")
    st.caption(
        "Every candidate, and the gate that removed it. Rules run before the model: "
        "no score can promote a candidate past eligibility, consent, or the contact cap."
    )
    for line in decision.trace:
        st.code(line, language=None)


# -------------------------------------------------------------------- ops view

with ops_tab:
    dec, compliance = run_all(min_prob, max_contacts)
    acted = dec[dec["contacted"]]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Customers scored", f"{len(dec):,}")
    k2.metric("Contacted", f"{len(acted):,}")
    k3.metric("Suppression rate", f"{1 - len(acted)/len(dec):.1%}")
    k4.metric("Consent violations", compliance["violations"],
              delta="must be 0", delta_color="off")

    if compliance["violations"] == 0:
        st.success("Suppression compliance: 100% — no action sent without a consented channel.")
    else:
        st.error(f"{compliance['violations']} consent violations — this blocks launch.")

    st.divider()
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Action mix")
        mix = acted["action"].value_counts()
        mix.index = [LABELS.get(i, i) for i in mix.index]
        st.bar_chart(mix)
        top = mix.iloc[0] / mix.sum() if len(mix) else 0
        if top > 0.5:
            st.warning(
                f"Diversity guardrail: **{mix.index[0]}** is {top:.0%} of all contacts. "
                "Expected-value arbitration concentrates on the most predictable action — "
                "probability spread across products (~100x) exceeds value spread (~5x). "
                "An explicit diversity mechanism is a phase-2 item."
            )

    with c2:
        st.subheader("Channel mix")
        ch = acted["channel"].value_counts()
        ch.index = [CHANNEL_LABELS.get(i, i) for i in ch.index]
        st.bar_chart(ch)

    st.subheader("Why customers were suppressed")
    REASON_SHORT = {
        "No eligible, consented action clears the value floor.": "Below value floor",
        "Contact cap reached for this period.": "Contact cap reached",
        "Best available action has non-positive expected value.": "Negative expected value",
    }
    sup = (dec.loc[~dec["contacted"], "reason"].map(REASON_SHORT)
             .value_counts().rename("customers").to_frame())
    st.bar_chart(sup, horizontal=True)
    #this didn't work
    # st.bar_chart(dec.loc[~dec["contacted"], "reason"].value_counts())

    st.divider()
    st.subheader("Fairness audit")
    st.caption(
        "Protected attributes are excluded from model features and measured on outcomes. "
        "Exposure = share contacted. Precision = share of contacts where the customer "
        "adopted the recommended product."
    )

    truth = (P[P["added"] == 1].groupby("ncodpers")["product"].apply(set))
    aud = dec.join(ATTRS[["age", "sexo", "renta"]], how="inner")
    aud = aud[aud["age"] >= 18]
    aud["hit"] = [
        (a in truth.get(i, set())) if c and isinstance(a, str) else np.nan
        for i, c, a in zip(aud.index, aud["contacted"], aud["action"])
    ]
    aud["Age band"] = pd.cut(aud["age"], [17, 30, 45, 60, 200],
                             labels=["18–30", "31–45", "46–60", "60+"])
    aud["Income quartile"] = pd.qcut(aud["renta"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    aud["Sex"] = aud["sexo"].map({"H": "Group A", "V": "Group B"}).fillna("Unknown")

    for attr in ["Age band", "Income quartile", "Sex"]:
        g = aud.groupby(attr, observed=True).agg(
            exposure=("contacted", "mean"),
            contacted_n=("contacted", "sum"),
            precision=("hit", "mean"),
        )
        ratio_e = g["exposure"].max() / g["exposure"].min()
        ratio_p = g["precision"].max() / g["precision"].min()
        st.markdown(f"**{attr}** — exposure ratio {ratio_e:.2f}× · precision ratio {ratio_p:.2f}×")
        st.dataframe(
            g.style.format({"exposure": "{:.1%}", "precision": "{:.2%}",
                            "contacted_n": "{:,.0f}"}),
            use_container_width=True,
        )

    st.info(
        "Exposure and outcome parity diverge. Differential exposure can reflect genuine "
        "differences in propensity. A precision gap cannot — it means some groups receive "
        "less accurate recommendations when they are contacted. Age was excluded from "
        "features; tenure and the activity flag act as proxies. Measuring the attribute "
        "is what makes the residual visible."
    )