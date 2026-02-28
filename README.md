# 🚀 Conversational Data Intelligence Platform

An AI-powered, secure Conversational Data Intelligence Platform that enables users to query structured CSV datasets using natural language and receive explainable, data-backed insights.

Built using **Flask + Pandas + LLM (Groq) + HTML/CSS/JS**, the system translates business questions into executable pandas code, runs them securely server-side, and returns insights with transparency metrics and visualizations.

---

## 🎯 Problem Statement

Modern businesses generate large volumes of structured data, but non-technical stakeholders struggle to extract insights without relying on data teams.

This platform solves that problem by enabling:

- Natural language querying of structured datasets
- Secure execution of generated analytics code
- Explainable and transparent results
- Dataset-grounded answers (no hallucinations)

---

## 🧠 Key Features

### ✅ Natural Language to Pandas
- Converts user questions into executable pandas code using LLM
- Deterministic and schema-grounded query generation

### 🔒 Secure Server-Side Execution
- Code runs in a restricted sandbox environment
- No file access, no network access, no unsafe operations
- Only approved built-in functions allowed

### 📊 Data Preview
- Preview top 10 rows of uploaded CSV
- Dynamic table rendering inside modal

### 📁 Data Passport
- Total rows & columns
- Column types
- Unique values & samples

### 📈 Auto Visualization
- Automatically generates charts for grouped results
- Bar charts for categorical comparisons

### 🛡 Confidence & Transparency
- Columns used
- Rows processed
- Analysis type
- Complexity score
- Execution time
- Confidence score
- External data usage (always False)

### 📝 Explainable Outputs
- Business-friendly summary
- Technical explanation of how result was derived
- Executed query logic shown transparently

---

## 🔐 Security Model

- Server-side execution (Flask backend)
- Restricted `eval()` namespace
- No imports, OS commands, or file operations allowed
- Dataset-grounded analysis only
- No external data access
- No predictive or unsupported analysis

---

## 🛠 Tech Stack

- **Backend:** Flask (Python)
- **Data Processing:** Pandas
- **LLM:** Llama model via Groq API
- **Frontend:** HTML, CSS, JavaScript
- **Visualization:** Chart.js
- **Security:** Restricted execution sandbox
