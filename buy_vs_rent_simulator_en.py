#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Buy vs Rent Financial Simulator — English UI
- Streamlit app to compare long-term wealth between buying vs renting.
- Includes: salary & expense growth, mortgage amortization, investments (bank/stock/crypto),
  salary DCA, rent escalation, home appreciation, annual/monthly views, and
  yearly inflow/outflow stacked breakdown.
"""

import json
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Any, List

import pandas as pd
import streamlit as st
import altair as alt

# =====================
# Core simulator
# =====================

@dataclass
class Params:
    # Income & tax
    annual_salary: float = 240000.0
    ca_tax_rate: float = 0.09
    salary_annual_growth: float = 0.03
    expense_annual_growth: float = 0.02

    # Bank & investments
    bank_initial: float = 50000.0
    bank_annual_rate: float = 0.02
    stock_initial: float = 50000.0
    stock_annual_return: float = 0.07
    crypto_initial: float = 10000.0
    crypto_annual_return: float = 0.15

    # Monthly DCA from net salary to stocks
    salary_stock_pct: float = 0.10

    # Housing (buy)
    home_price: float = 1_000_000.0
    mortgage_principal: float = 800_000.0
    mortgage_annual_rate: float = 0.055
    mortgage_term_years: int = 30

    # Common housing costs (buy)
    property_tax_rate: float = 0.011
    maintenance_rate: float = 0.01
    hoa_monthly: float = 0.0
    home_ins_monthly: float = 0.0

    # Rent
    rent_monthly: float = 3500.0
    rent_annual_growth: float = 0.03

    # Other expenses
    monthly_expenses: float = 4000.0

    # Home appreciation & sale costs
    home_growth: float = 0.03
    sell_cost_rate: float = 0.06

    # Horizon
    years: int = 10

    # Optional: cash equalization on rent side (inject down payment into investments)
    MATCH_CASH_OUT: bool = False
    rent_inject_split_stock: float = 0.7
    rent_inject_split_crypto: float = 0.3


def mrate(annual: float) -> float:
    """Convert annual rate to effective monthly rate (compounded)."""
    return (1.0 + annual) ** (1.0 / 12.0) - 1.0


def monthly_payment(principal: float, annual_rate: float, term_years: int) -> float:
    """Standard fixed-rate mortgage monthly payment (amortized)."""
    n = term_years * 12
    r = mrate(annual_rate)
    if r == 0:
        return principal / n
    return (principal * r) / (1.0 - (1.0 + r) ** (-n))


def simulate(p: Params) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    months = p.years * 12
    gross = p.annual_salary / 12.0
    net_income = gross * (1.0 - p.ca_tax_rate)

    bank_mr = mrate(p.bank_annual_rate)
    stock_mr = mrate(p.stock_annual_return)
    crypto_mr = mrate(p.crypto_annual_return)
    loan_mr = mrate(p.mortgage_annual_rate)

    pay = monthly_payment(p.mortgage_principal, p.mortgage_annual_rate, p.mortgage_term_years)
    tax_m = (p.home_price * p.property_tax_rate) / 12.0
    maint_m = (p.home_price * p.maintenance_rate) / 12.0
    hoa_ins_m = p.hoa_monthly + p.home_ins_monthly

    rent_growth_mr = mrate(p.rent_annual_growth)
    rent_cur = p.rent_monthly

    down = max(0.0, p.home_price - p.mortgage_principal)
    if down > p.bank_initial:
        raise ValueError(
            f"Down payment shortfall: need ${down:,.0f}, but bank has only ${p.bank_initial:,.0f}."
        )

    # For summary display
    bank_initial_after_buy = p.bank_initial - down
    bank_initial_rent = p.bank_initial

    # BUY initial balances
    bank_b = bank_initial_after_buy
    stock_b = p.stock_initial
    crypto_b = p.crypto_initial

    # RENT initial balances
    bank_r = p.bank_initial
    stock_r = p.stock_initial
    crypto_r = p.crypto_initial
    if p.MATCH_CASH_OUT and down > 0:
        stock_r += down * p.rent_inject_split_stock
        crypto_r += down * p.rent_inject_split_crypto
        rest = 1.0 - (p.rent_inject_split_stock + p.rent_inject_split_crypto)
        if rest > 0:
            bank_r += down * rest

    remaining = p.mortgage_principal
    home_val = p.home_price
    expense_cur = p.monthly_expenses

    # logs
    m_rows: List[Dict[str, Any]] = []
    y_rows: List[Dict[str, Any]] = []
    y_breakdown: List[Dict[str, Any]] = []

    # yearly accumulators (housing)
    y_mtg_paid = y_interest = y_principal = 0.0
    y_tax = y_maint = y_hoains = 0.0
    y_rent_paid = 0.0

    # yearly investment returns (accumulated monthly)
    y_bank_ret_buy = y_stock_ret_buy = y_crypto_ret_buy = 0.0
    y_bank_ret_rent = y_stock_ret_rent = y_crypto_ret_rent = 0.0
    y_home_app = 0.0

    # yearly flows for breakdown
    y_stock_contrib = 0.0      # DCA from salary (cash outflow)
    y_salary_net = 0.0         # Net salary inflow (after tax)
    y_expenses_total = 0.0     # Living expenses
    y_downpayment_buy = down   # One-time in year 1

    for m in range(1, months + 1):
        # Net salary this month
        y_salary_net += net_income

        # DCA this month
        stock_contrib = net_income * p.salary_stock_pct
        y_stock_contrib += stock_contrib
        net_after_contrib = net_income - stock_contrib

        # Living expenses
        y_expenses_total += expense_cur

        # ===== BUY =====
        mtg = min(pay, remaining + pay)
        interest = remaining * loan_mr
        principal = max(0.0, mtg - interest)
        remaining = max(0.0, remaining - principal)
        housing_b = mtg + tax_m + maint_m + hoa_ins_m

        stock_b_before = stock_b
        stock_b += stock_contrib
        stock_b *= (1 + stock_mr)
        y_stock_ret_buy += (stock_b - stock_b_before - stock_contrib)

        bank_b += net_after_contrib - expense_cur - housing_b
        if bank_b > 0:
            bank_interest = bank_b * bank_mr
            bank_b *= (1 + bank_mr)
            y_bank_ret_buy += bank_interest

        crypto_before = crypto_b
        crypto_b *= (1 + crypto_mr)
        y_crypto_ret_buy += (crypto_b - crypto_before)

        # ===== RENT =====
        housing_r = rent_cur

        stock_r_before = stock_r
        stock_r += stock_contrib
        stock_r *= (1 + stock_mr)
        y_stock_ret_rent += (stock_r - stock_r_before - stock_contrib)

        bank_r += net_after_contrib - expense_cur - housing_r
        if bank_r > 0:
            bank_interest_r = bank_r * bank_mr
            bank_r *= (1 + bank_mr)
            y_bank_ret_rent += bank_interest_r

        crypto_r_before = crypto_r
        crypto_r *= (1 + crypto_mr)
        y_crypto_ret_rent += (crypto_r - crypto_r_before)

        # Monthly logs
        m_rows.append({
            "month": m,
            "year": (m - 1) // 12 + 1,
            "buy_housing": housing_b,
            "rent_housing": housing_r,
            "buy_total_monthly_outflow": housing_b + expense_cur + stock_contrib,
            "rent_total_monthly_outflow": housing_r + expense_cur + stock_contrib,
            "buy_net_cashflow": net_after_contrib - (housing_b + expense_cur),
            "rent_net_cashflow": net_after_contrib - (housing_r + expense_cur),
            "stock_contrib": stock_contrib,
        })

        # Yearly accumulators (housing)
        y_mtg_paid += mtg
        y_interest += interest
        y_principal += principal
        y_tax += tax_m
        y_maint += maint_m
        y_hoains += hoa_ins_m
        y_rent_paid += housing_r

        # Home appreciation (accumulate monthly amount)
        home_app_m = home_val * (p.home_growth / 12.0)
        y_home_app += home_app_m
        home_val *= (1 + p.home_growth / 12.0)

        # Evolve rent
        rent_cur *= (1 + rent_growth_mr)

        # Year-end close
        if m % 12 == 0:
            year = m // 12
            equity = home_val - remaining
            sale_proceeds = home_val - home_val * p.sell_cost_rate - remaining
            if sale_proceeds < 0:
                sale_proceeds = 0.0

            total_buy = bank_b + stock_b + crypto_b + sale_proceeds
            total_rent = bank_r + stock_r + crypto_r

            # Annual master table
            y_rows.append({
                "year": year,
                "buy_total": total_buy,
                "rent_total": total_rent,
                "diff_buy_minus_rent": total_buy - total_rent,
                "buy_bank": bank_b, "buy_stock": stock_b, "buy_crypto": crypto_b,
                "home_value": home_val, "home_equity": equity,
                "remaining_principal": remaining, "sale_proceeds_if_sold": sale_proceeds,
                "buy_housing_total": y_mtg_paid + y_tax + y_maint + y_hoains,
                "buy_mortgage_paid": y_mtg_paid, "buy_interest": y_interest,
                "buy_principal": y_principal, "buy_property_tax": y_tax,
                "buy_maintenance": y_maint, "buy_hoa_ins": y_hoains,
                "rent_bank": bank_r, "rent_stock": stock_r, "rent_crypto": crypto_r,
                "rent_paid": y_rent_paid,
                "stock_contrib_year": y_stock_contrib,
            })

            # Annual inflow/outflow breakdown (fully specified)
            invest_ret_buy = y_bank_ret_buy + y_stock_ret_buy + y_crypto_ret_buy
            invest_ret_rent = y_bank_ret_rent + y_stock_ret_rent + y_crypto_ret_rent

            # Buy scenario breakdown
            y_breakdown.append({
                "year": year, "scenario": "Buy",
                "Net Salary": y_salary_net,
                "DCA from Salary": -y_stock_contrib,
                "Living Expenses": -y_expenses_total,
                "Principal Repaid": y_principal,
                "Home Appreciation": y_home_app,
                "Investment Returns": invest_ret_buy,
                "Mortgage Interest": -y_interest,
                "Property Tax": -y_tax,
                "Maintenance": -y_maint,
                "HOA/Insurance": -y_hoains,
                "Rent": 0.0,
                "Down Payment": -y_downpayment_buy,  # non-zero only in Year 1
            })

            # Rent scenario breakdown
            y_breakdown.append({
                "year": year, "scenario": "Rent",
                "Net Salary": y_salary_net,
                "DCA from Salary": -y_stock_contrib,
                "Living Expenses": -y_expenses_total,
                "Principal Repaid": 0.0,
                "Home Appreciation": 0.0,
                "Investment Returns": invest_ret_rent,
                "Mortgage Interest": 0.0,
                "Property Tax": 0.0,
                "Maintenance": 0.0,
                "HOA/Insurance": 0.0,
                "Rent": -y_rent_paid,
                "Down Payment": 0.0,
            })

            # Update salary & expenses for next year
            gross *= (1 + p.salary_annual_growth)
            net_income = gross * (1 - p.ca_tax_rate)
            expense_cur *= (1 + p.expense_annual_growth)

            # Reset yearly accumulators
            y_mtg_paid = y_interest = y_principal = 0.0
            y_tax = y_maint = y_hoains = 0.0
            y_rent_paid = 0.0
            y_bank_ret_buy = y_stock_ret_buy = y_crypto_ret_buy = 0.0
            y_bank_ret_rent = y_stock_ret_rent = y_crypto_ret_rent = 0.0
            y_home_app = 0.0
            y_stock_contrib = 0.0
            y_salary_net = 0.0
            y_expenses_total = 0.0
            y_downpayment_buy = 0.0  # only counted in first year

    df_yearly = pd.DataFrame(y_rows)
    df_monthly = pd.DataFrame(m_rows)
    df_break = pd.DataFrame(y_breakdown)

    meta = {
        "monthly_payment": pay,
        "down_payment": down,
        "params": asdict(p),
        "monthly_net_income_start": (p.annual_salary / 12.0) * (1.0 - p.ca_tax_rate),
        "bank_initial_after_buy": bank_initial_after_buy,
        "bank_initial_rent": bank_initial_rent,
    }
    return df_yearly, meta, df_monthly, df_break


def fm(x: float) -> str:
    return f"${x:,.0f}"


# =====================
# Streamlit UI — English
# =====================

st.set_page_config(page_title="Buy vs Rent — 10-Year Wealth Simulator", layout="wide")
st.title("Buy vs Rent: 10-Year Household Wealth Simulator (Dual Income)")

with st.sidebar:
    st.header("Income & Taxes")
    annual_salary = st.number_input("Annual Combined Salary ($)", value=240000, step=1000)
    ca_tax_rate = st.number_input("Effective Tax Rate (%)", value=9.0, step=0.1) / 100.0
    salary_growth = st.number_input("Salary Annual Growth (%)", value=3.0, step=0.1) / 100.0
    expense_growth = st.number_input("Expense Annual Growth (%)", value=2.0, step=0.1) / 100.0

    st.header("Cash & Investments")
    bank_initial = st.number_input("Initial Bank Savings ($)", value=50000, step=1000)
    bank_rate = st.number_input("Bank Annual Interest (%)", value=2.0, step=0.1) / 100.0
    stock_initial = st.number_input("Initial Stock Balance ($)", value=50000, step=1000)
    stock_return = st.number_input("Stock Annual Return (%)", value=7.0, step=0.1) / 100.0
    crypto_initial = st.number_input("Initial Crypto Balance ($)", value=10000, step=1000)
    crypto_return = st.number_input("Crypto Annual Return (%)", value=15.0, step=0.1) / 100.0
    salary_stock_pct = st.slider("Monthly DCA from Salary to Stocks (%)", 0, 80, 10) / 100.0

    st.header("Buy Scenario — Mortgage & Home")
    home_price = st.number_input("Home Price ($)", value=1_000_000, step=5000)
    mortgage_principal = st.number_input("Mortgage Principal ($)", value=800_000, step=5000)
    mortgage_rate = st.number_input("Mortgage Annual Rate (%)", value=5.5, step=0.1) / 100.0
    term_years = st.number_input("Mortgage Term (years)", value=30, step=1)
    property_tax_rate = st.number_input("Property Tax (% of price / year)", value=1.1, step=0.05) / 100.0
    maintenance_rate = st.number_input("Maintenance (% of price / year)", value=1.0, step=0.05) / 100.0
    hoa_monthly = st.number_input("HOA ($/month)", value=0, step=25)
    home_ins_monthly = st.number_input("Home Insurance ($/month)", value=0, step=10)

    st.header("Rent Scenario")
    rent_monthly = st.number_input("Initial Monthly Rent ($)", value=3500, step=100)
    rent_growth = st.number_input("Rent Annual Growth (%)", value=3.0, step=0.1) / 100.0

    st.header("Other Settings")
    monthly_expenses = st.number_input("Initial Monthly Living Expenses ($)", value=4000, step=100)
    home_growth = st.number_input("Home Annual Appreciation (%)", value=3.0, step=0.1) / 100.0
    sell_cost_rate = st.number_input("Selling Transaction Cost (% of price)", value=6.0, step=0.1) / 100.0
    years = st.number_input("Simulation Horizon (years)", value=10, step=1)

    st.divider()
    MATCH_CASH_OUT = st.checkbox("Cash-Equalized Rent (inject down payment into investments)", value=False)
    col = st.columns(2)
    with col[0]:
        inject_stock = st.number_input("Injection Split: Stocks (%)", value=70.0, step=1.0) / 100.0
    with col[1]:
        inject_crypto = st.number_input("Injection Split: Crypto (%)", value=30.0, step=1.0) / 100.0

    run_btn = st.button("Run Simulation")

if run_btn:
    p = Params(
        annual_salary=annual_salary, ca_tax_rate=ca_tax_rate,
        salary_annual_growth=salary_growth, expense_annual_growth=expense_growth,
        bank_initial=bank_initial, bank_annual_rate=bank_rate,
        stock_initial=stock_initial, stock_annual_return=stock_return,
        crypto_initial=crypto_initial, crypto_annual_return=crypto_return,
        salary_stock_pct=salary_stock_pct,
        home_price=home_price, mortgage_principal=mortgage_principal,
        mortgage_annual_rate=mortgage_rate, mortgage_term_years=int(term_years),
        property_tax_rate=property_tax_rate, maintenance_rate=maintenance_rate,
        hoa_monthly=hoa_monthly, home_ins_monthly=home_ins_monthly,
        rent_monthly=rent_monthly, rent_annual_growth=rent_growth,
        monthly_expenses=monthly_expenses, home_growth=home_growth,
        sell_cost_rate=sell_cost_rate, years=int(years),
        MATCH_CASH_OUT=MATCH_CASH_OUT,
        rent_inject_split_stock=inject_stock, rent_inject_split_crypto=inject_crypto,
    )

    try:
        df_y, meta, df_m, df_break = simulate(p)
    except ValueError as e:
        st.error(str(e))
    else:
        st.subheader("Summary")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: st.metric("Monthly Mortgage Payment", fm(meta["monthly_payment"]))
        with c2: st.metric("Down Payment", fm(meta["down_payment"]))
        with c3: st.metric("Initial Bank (Buy, after DP)", fm(meta["bank_initial_after_buy"]))
        with c4: st.metric("Initial Bank (Rent)", fm(meta["bank_initial_rent"]))
        with c5: st.metric("Starting Net Monthly Income", fm(meta["monthly_net_income_start"]))

        st.subheader("Annual Total Wealth (Buy vs Rent)")
        st.line_chart(df_y.set_index("year")[ ["buy_total", "rent_total"] ])

        st.subheader("Investment Balances by Scenario (Annual)")
        colA, colB = st.columns(2)
        with colA:
            st.caption("Buy: Bank / Stocks / Crypto (ex-home)")
            st.line_chart(df_y.set_index("year")[ ["buy_bank", "buy_stock", "buy_crypto"] ])
        with colB:
            st.caption("Rent: Bank / Stocks / Crypto")
            st.line_chart(df_y.set_index("year")[ ["rent_bank", "rent_stock", "rent_crypto"] ])

        st.subheader("Home Metrics (Annual)")
        st.line_chart(df_y.set_index("year")[ ["home_value", "home_equity", "remaining_principal"] ])

        st.subheader("Monthly Costs & Net Cash Flow (Buy vs Rent)")
        tab1, tab2, tab3, tab4 = st.tabs(["Housing Cost Only (Monthly)", "Total Outflow (Monthly)", "Net Cash Flow (Monthly)", "Monthly DCA Amount"])
        with tab1:
            st.line_chart(df_m.set_index("month")[ ["buy_housing", "rent_housing"] ])
        with tab2:
            m2 = df_m.set_index("month")[ ["buy_total_monthly_outflow", "rent_total_monthly_outflow"] ]
            m2.columns = ["Buy: Total Monthly Outflow", "Rent: Total Monthly Outflow"]
            st.line_chart(m2)
        with tab3:
            m3 = df_m.set_index("month")[ ["buy_net_cashflow", "rent_net_cashflow"] ]
            m3.columns = ["Buy: Net Cash Flow", "Rent: Net Cash Flow"]
            st.line_chart(m3)
        with tab4:
            st.line_chart(df_m.set_index("month")[ ["stock_contrib"] ])

        # Annual inflow/outflow stacked bars
        st.subheader("Annual Asset Inflows/Outflows (Stacked)")
        df_long = df_break.melt(id_vars=["year", "scenario"], var_name="item", value_name="amount")
        order_items = [
            "Net Salary", "Investment Returns", "Principal Repaid", "Home Appreciation",
            "DCA from Salary", "Living Expenses", "Mortgage Interest", "Property Tax", "Maintenance", "HOA/Insurance", "Rent", "Down Payment"
        ]
        chart = (
            alt.Chart(df_long)
            .mark_bar()
            .encode(
                x=alt.X("year:O", title="Year"),
                y=alt.Y("sum(amount):Q", title="Annual Amount ($)"),
                color=alt.Color("item:N", sort=order_items),
                column=alt.Column("scenario:N", title="", sort=["Buy", "Rent"]),
                tooltip=["scenario", "year", "item", alt.Tooltip("amount:Q", format=",")]
            )
            .properties(height=320)
        )
        st.altair_chart(chart, use_container_width=True)

        # Liquidity warning
        neg_b = int((df_m["buy_net_cashflow"] < 0).sum())
        neg_r = int((df_m["rent_net_cashflow"] < 0).sum())
        if neg_b or neg_r:
            st.warning(f"Negative monthly net cash flow months detected — Buy: {neg_b} months; Rent: {neg_r} months. Mind liquidity risk.")

        st.subheader("Download Data")
        st.download_button("Download Annual CSV", data=df_y.to_csv(index=False), file_name="buy_vs_rent_yearly.csv", mime="text/csv")
        st.download_button("Download Monthly CSV", data=df_m.to_csv(index=False), file_name="buy_vs_rent_monthly.csv", mime="text/csv")
        st.download_button("Download Breakdown CSV", data=df_break.to_csv(index=False), file_name="buy_vs_rent_breakdown.csv", mime="text/csv")
        st.download_button("Download Params JSON", data=json.dumps(p.__dict__, ensure_ascii=False, indent=2), file_name="params.json", mime="application/json")
else:
    st.info("Set parameters on the left and click 'Run Simulation'.")
