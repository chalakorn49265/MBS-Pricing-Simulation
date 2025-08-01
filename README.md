# 🏦 MBS Pricing Simulator

This project implements a Monte Carlo simulation framework to price Mortgage-Backed Securities (MBS) using the CIR interest rate model and a Numerix-style prepayment model. It includes functionalities to compute MBS price, Option-Adjusted Spread (OAS), and to decompose and price IO and PO tranches.

## 📘 Overview

- **Interest Rate Model**: Cox-Ingersoll-Ross (CIR)
- **Prepayment Model**: Numerix-style (Burnout, Refinancing Incentive, Seasonality, Ramp-up)
- **Pricing Method**: Monte Carlo Simulation (10,000 paths)
- **Outputs**:
  - MBS Fair Value
  - Option-Adjusted Spread (OAS)
  - IO/PO Tranche Pricing
  - Scenario Analysis over long-term mean rate (\( \bar{r} \))

---

## 🧠 Key Features

- 📈 **CIR Rate Simulation**: Captures mean-reverting, non-negative short rate dynamics
- 🔁 **Prepayment Modeling**: Incorporates CPR as a function of refinancing incentives, balance burnout, and seasonal factors
- ⚖️ **OAS Calculation**: Root-finding approach to match market price
- 🔍 **Tranche Decomposition**: Computes present value of Interest-Only and Principal-Only cash flows
- 📊 **Scenario Analysis**: Visualizes MBS price sensitivity to changes in \( \bar{r} \)

---

## 🚀 How to Run

### Prerequisites
```bash
pip install numpy matplotlib scipy
